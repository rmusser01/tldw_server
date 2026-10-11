import React from "react"
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ChatWorkspacePage } from "../ChatWorkspacePage"

const setRouteContext = vi.fn()
const chatPanelRuntimeState = vi.hoisted(() => ({
  backendAvailable: true,
  streaming: true,
  sending: false,
  historyLoading: false,
  historyLoadError: null as string | null
}))
const workspaceState = vi.hoisted((): { value: any } => ({
  value: {
    workspaceId: "workspace-1",
    storeHydrated: true,
    workspaceName: "Default workspace",
    sources: [
      {
        id: "source-1",
        mediaId: 101,
        title: "Operator Notes",
        type: "document",
        status: "ready",
        addedAt: new Date("2026-05-03T00:00:00Z")
      }
    ],
    effectiveAssistantDefault: {
      status: "available",
      source: "workspace",
      assistantKind: "persona",
      assistantId: "workspace-persona",
      label: "Workspace Analyst",
      personaMemoryMode: "read_only",
      degradedReason: null
    }
  }
}))
const workspaceActions = vi.hoisted(() => ({
  focusSourceById: vi.fn(() => true),
  initializeWorkspace: vi.fn(() => "workspace-first")
}))
const rehydrateWorkspace = vi.hoisted(() => vi.fn())
const connectionState = vi.hoisted(() => ({
  value: {
    phase: "connected",
    isConnected: true,
    serverUrl: "http://127.0.0.1:8000",
    mode: "normal",
    offlineBypass: false
  }
}))
const chatPanelClearHandlers = vi.hoisted(() => new Map<string, () => void>())
const chatPanelRemoveHandlers = vi.hoisted(
  () => new Map<string, (sourceId: string) => void>()
)
const previewCloseHandlers = vi.hoisted(() => new Map<string, () => void>())

vi.mock("../../ResearchWorkspace/ResearchWorkspaceRouteGate", () => ({
  ActivatedLocalWorkspace: ({ children, workspaceId }: { children: React.ReactNode; workspaceId: string }) => (
    <section data-testid="workspace-activation" data-workspace-id={workspaceId}>{children}</section>
  )
}))

vi.mock("../../ResearchWorkspace/SourcesPane/WorkspaceSourcePreview", () => ({
  WorkspaceSourcePreview: ({
    workspaceId,
    source,
    onClose
  }: {
    workspaceId: string | null
    source: { id: string; title: string } | null
    onClose: () => void
  }) => {
    if (!source) return null
    previewCloseHandlers.set(`${workspaceId}:${source.id}`, onClose)
    return (
      <section role="dialog" aria-label={`Preview ${source.title}`}>
        <button type="button" onClick={onClose}>
          Close preview
        </button>
      </section>
    )
  }
}))

vi.mock("@/store/chat-surface-coordinator", () => ({
  useChatSurfaceCoordinatorStore: (selector: any) =>
    selector({ setRouteContext })
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: Object.assign(
    (selector: any) =>
      selector({
        sourcesLoading: false,
        sourcesError: null,
        storeHydrated: true,
        serverWorkspace: { metadata: { id: workspaceState.value.workspaceId } },
        ...workspaceActions,
        ...workspaceState.value
      }),
    {
      persist: { rehydrate: rehydrateWorkspace },
      getState: () => ({
        ...workspaceActions,
        ...workspaceState.value
      })
    }
  )
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => connectionState.value
}))

vi.mock("../WorkspaceChatPanel", () => ({
  WorkspaceChatPanel: ({
    stagedSources,
    workspaceId,
    onClearStagedSources,
    onRemoveStagedSource,
    backendAvailable,
    effectiveAssistantDefault,
    onRuntimeStateChange
  }: {
    stagedSources: unknown[]
    workspaceId?: string | null
    onClearStagedSources: () => void
    onRemoveStagedSource?: (sourceId: string) => void
    backendAvailable: boolean
    effectiveAssistantDefault?: { assistantId?: string | null } | null
    onRuntimeStateChange?: (state: unknown) => void
  }) => {
    const [mountedWorkspaceId] = React.useState(workspaceId)

    if (workspaceId) {
      chatPanelClearHandlers.set(workspaceId, onClearStagedSources)
      if (onRemoveStagedSource) {
        chatPanelRemoveHandlers.set(workspaceId, onRemoveStagedSource)
      }
    }

    React.useEffect(() => {
      onRuntimeStateChange?.({
        backendAvailable: chatPanelRuntimeState.backendAvailable,
        streaming: chatPanelRuntimeState.streaming,
        sending: chatPanelRuntimeState.sending,
        historyLoading: chatPanelRuntimeState.historyLoading,
        historyLoadError: chatPanelRuntimeState.historyLoadError,
        selectedModelLabel: "gpt-test",
        hasModelSelected: true,
        selectedPersonaLabel: "Analyst",
        assistantSource: "explicit"
      })
    }, [onRuntimeStateChange])

    return (
      <section
        data-testid="workspace-chat-panel"
        data-workspace-id={workspaceId ?? "null"}
        data-backend-available={String(backendAvailable)}
        data-effective-assistant-id={
          effectiveAssistantDefault?.assistantId ?? "null"
        }>
        staged:{stagedSources.length}; workspace:{workspaceId}; mounted:
        {mountedWorkspaceId}; backend:{String(backendAvailable)}
      </section>
    )
  }
}))

describe("ChatWorkspacePage", () => {
  it("uses the shell main landmark without nesting another main", () => {
    render(
      <main>
        <ChatWorkspacePage />
      </main>
    )
    expect(screen.getAllByRole("main")).toHaveLength(1)
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1)
  })

  it("selects narrow-screen panes without remounting the workspace chat", () => {
    render(<ChatWorkspacePage />)
    const panels = screen.getByRole("navigation", { name: "Workspace panels" })
    const chat = within(panels).getByRole("button", {
      name: "Chat",
      exact: true
    })
    const sources = within(panels).getByRole("button", {
      name: "Sources",
      exact: true
    })
    const mountedPanel = screen.getByTestId("workspace-chat-panel")
    expect(chat).toHaveAttribute("aria-pressed", "true")
    fireEvent.click(sources)
    expect(sources).toHaveAttribute("aria-pressed", "true")
    expect(chat).toHaveAttribute("aria-pressed", "false")
    expect(
      document.getElementById(sources.getAttribute("aria-controls")!)
    ).toContainElement(
      screen.getByRole("complementary", { name: /workspace sources/i })
    )
    fireEvent.click(chat)
    expect(screen.getByTestId("workspace-chat-panel")).toBe(mountedPanel)
  })

  beforeEach(() => {
    vi.clearAllMocks()
    rehydrateWorkspace.mockResolvedValue(undefined)
    window.localStorage.clear()
    delete (
      window as Window & {
        __tldwResearchWorkspaceFreshInitialization?: Set<string>
      }
    ).__tldwResearchWorkspaceFreshInitialization
    chatPanelRuntimeState.backendAvailable = true
    chatPanelRuntimeState.streaming = true
    chatPanelRuntimeState.sending = false
    chatPanelRuntimeState.historyLoading = false
    chatPanelRuntimeState.historyLoadError = null
    workspaceActions.focusSourceById.mockReturnValue(true)
    workspaceActions.initializeWorkspace.mockImplementation(() => {
      workspaceState.value = {
        ...workspaceState.value,
        workspaceId: "workspace-first"
      }
      return "workspace-first"
    })
    workspaceState.value = {
      workspaceId: "workspace-1",
      storeHydrated: true,
      workspaceName: "Default workspace",
      sources: [
        {
          id: "source-1",
          mediaId: 101,
          title: "Operator Notes",
          type: "document",
          status: "ready",
          addedAt: new Date("2026-05-03T00:00:00Z")
        }
      ],
      effectiveAssistantDefault: {
        status: "available",
        source: "workspace",
        assistantKind: "persona",
        assistantId: "workspace-persona",
        label: "Workspace Analyst",
        personaMemoryMode: "read_only",
        degradedReason: null
      }
    }
    connectionState.value = {
      phase: "connected",
      isConnected: true,
      serverUrl: "http://127.0.0.1:8000",
      mode: "normal",
      offlineBypass: false
    }
    chatPanelClearHandlers.clear()
    chatPanelRemoveHandlers.clear()
    previewCloseHandlers.clear()
  })

  it.each([
    { mode: "demo", offlineBypass: false, label: "Demo mode - not live" },
    { mode: "normal", offlineBypass: true, label: "Offline bypass - not verified" }
  ])("distinguishes synthetic connection from live readiness (%j)", ({ mode, offlineBypass, label }) => {
    connectionState.value = { ...connectionState.value, mode, offlineBypass }
    render(<ChatWorkspacePage />)
    expect(screen.getByRole("status")).toHaveTextContent(label)
    expect(screen.getByTestId("workspace-chat-panel")).toHaveAttribute("data-backend-available", "false")
  })

  it.each([
    [{ sending: true }, "Sending"],
    [{ historyLoading: true }, "Loading chat history"],
    [{ historyLoadError: "History unavailable" }, "Chat history unavailable"]
  ] as const)("propagates panel runtime to both status surfaces (%j)", (runtime, label) => {
    Object.assign(chatPanelRuntimeState, { streaming: false }, runtime)
    render(<ChatWorkspacePage />)
    expect(screen.getByRole("status")).toHaveTextContent(label)
    expect(within(screen.getByRole("complementary", { name: "Chat workspace inspector" })).getByText(label)).toBeInTheDocument()
    expect(screen.queryByText("Ready")).not.toBeInTheDocument()
  })

  it("initializes an empty hydrated workspace once under StrictMode", async () => {
    workspaceState.value = { ...workspaceState.value, workspaceId: null }
    const mounted = render(
      <React.StrictMode>
        <ChatWorkspacePage />
      </React.StrictMode>
    )
    await waitFor(() =>
      expect(workspaceActions.initializeWorkspace).toHaveBeenCalledTimes(1)
    )
    mounted.rerender(
      <React.StrictMode>
        <ChatWorkspacePage />
      </React.StrictMode>
    )
    expect(screen.getByTestId("workspace-chat-panel")).toHaveAttribute(
      "data-workspace-id",
      "workspace-first"
    )
    expect(screen.getByTestId("workspace-chat-panel")).toHaveAttribute(
      "data-backend-available",
      "true"
    )
    expect(
      (
        window as Window & {
          __tldwResearchWorkspaceFreshInitialization?: Set<string>
        }
      ).__tldwResearchWorkspaceFreshInitialization?.has("workspace-first")
    ).toBe(true)
  })

  it("does not initialize until workspace persistence is hydrated", async () => {
    workspaceState.value = {
      ...workspaceState.value,
      workspaceId: null,
      storeHydrated: false
    }
    const mounted = render(<ChatWorkspacePage />)
    await act(async () => {})
    expect(workspaceActions.initializeWorkspace).not.toHaveBeenCalled()
    workspaceState.value = { ...workspaceState.value, storeHydrated: true }
    mounted.rerender(<ChatWorkspacePage />)
    await waitFor(() =>
      expect(workspaceActions.initializeWorkspace).toHaveBeenCalledTimes(1)
    )
  })

  it("does not replace an existing hydrated workspace", async () => {
    render(<ChatWorkspacePage />)
    await act(async () => {})
    expect(workspaceActions.initializeWorkspace).not.toHaveBeenCalled()
  })

  it("rereads an intervening active workspace before queued initialization", async () => {
    workspaceState.value = { ...workspaceState.value, workspaceId: null }
    render(<ChatWorkspacePage />)
    workspaceState.value = {
      ...workspaceState.value,
      workspaceId: "selected-before-initialization"
    }
    await act(async () => {})
    expect(workspaceActions.initializeWorkspace).not.toHaveBeenCalled()
  })

  it("cancels queued initialization when the page unmounts", async () => {
    workspaceState.value = { ...workspaceState.value, workspaceId: null }
    const mounted = render(<ChatWorkspacePage />)
    mounted.unmount()
    await act(async () => {})
    expect(workspaceActions.initializeWorkspace).not.toHaveBeenCalled()
  })

  it("does not mark legacy persisted content as a freshly created workspace", async () => {
    window.localStorage.setItem(
      "tldw-workspace",
      JSON.stringify({ workspaces: [{ id: "legacy" }] })
    )
    workspaceState.value = { ...workspaceState.value, workspaceId: null }
    render(<ChatWorkspacePage />)
    await waitFor(() =>
      expect(workspaceActions.initializeWorkspace).toHaveBeenCalledTimes(1)
    )
    expect(
      (
        window as Window & {
          __tldwResearchWorkspaceFreshInitialization?: Set<string>
        }
      ).__tldwResearchWorkspaceFreshInitialization?.has("workspace-first")
    ).toBeFalsy()
  })

  it("sets chat surface route context and renders the console regions", () => {
    render(<ChatWorkspacePage />)

    expect(setRouteContext).toHaveBeenCalledWith({
      routeId: "chat-workspace",
      surface: "webui"
    })
    expect(
      screen.getByRole("complementary", { name: /workspace sources/i })
    ).toBeInTheDocument()
    expect(screen.getByTestId("workspace-chat-panel")).toBeInTheDocument()
    expect(
      screen.getByRole("complementary", { name: /workspace inspector/i })
    ).toBeInTheDocument()
  })

  it("stages sources only through the explicit rail action", () => {
    render(<ChatWorkspacePage />)

    fireEvent.click(
      screen.getByRole("button", { name: "Browse Operator Notes" })
    )
    expect(workspaceActions.focusSourceById).toHaveBeenCalledWith("source-1")
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "staged:0"
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Stage Operator Notes for chat" })
    )
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "staged:1"
    )
  })

  it("closes and reopens Browse without staging or remounting chat", () => {
    render(<ChatWorkspacePage />)
    const chat = screen.getByTestId("workspace-chat-panel")
    fireEvent.click(
      screen.getByRole("button", { name: "Browse Operator Notes" })
    )
    expect(
      screen.getByRole("dialog", { name: "Preview Operator Notes" })
    ).toBeInTheDocument()
    expect(chat).toHaveTextContent("staged:0")
    fireEvent.click(screen.getByRole("button", { name: "Close preview" }))
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    fireEvent.click(
      screen.getByRole("button", { name: "Browse Operator Notes" })
    )
    expect(
      screen.getByRole("dialog", { name: "Preview Operator Notes" })
    ).toBeInTheDocument()
    expect(screen.getByTestId("workspace-chat-panel")).toBe(chat)
    expect(chat).toHaveTextContent("staged:0")
  })

  it("ignores an old close callback after browsing another source", () => {
    workspaceState.value = {
      ...workspaceState.value,
      sources: [
        ...workspaceState.value.sources,
        {
          ...workspaceState.value.sources[0],
          id: "source-2",
          title: "Other Notes"
        }
      ]
    }
    render(<ChatWorkspacePage />)
    fireEvent.click(
      screen.getByRole("button", { name: "Browse Operator Notes" })
    )
    const closeOldPreview = previewCloseHandlers.get("workspace-1:source-1")!
    fireEvent.click(screen.getByRole("button", { name: "Browse Other Notes" }))
    act(() => closeOldPreview())
    expect(
      screen.getByRole("dialog", { name: "Preview Other Notes" })
    ).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Close preview" }))
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("ignores an old close callback after changing workspaces", () => {
    const mounted = render(<ChatWorkspacePage />)
    fireEvent.click(
      screen.getByRole("button", { name: "Browse Operator Notes" })
    )
    const closeOldPreview = previewCloseHandlers.get("workspace-1:source-1")!
    workspaceState.value = {
      ...workspaceState.value,
      workspaceId: "workspace-2"
    }
    mounted.rerender(<ChatWorkspacePage />)
    fireEvent.click(
      screen.getByRole("button", { name: "Browse Operator Notes" })
    )
    act(() => closeOldPreview())
    expect(
      screen.getByRole("dialog", { name: "Preview Operator Notes" })
    ).toBeInTheDocument()
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "staged:0"
    )
  })

  it("does not browse sources before workspace hydration completes", () => {
    workspaceState.value = { ...workspaceState.value, storeHydrated: false }
    render(<ChatWorkspacePage />)
    fireEvent.click(
      screen.getByRole("button", { name: "Browse Operator Notes" })
    )
    expect(workspaceActions.focusSourceById).not.toHaveBeenCalled()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("passes workspace scope and real runtime state into the visible rails", async () => {
    render(<ChatWorkspacePage />)

    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "workspace:workspace-1"
    )
    expect(screen.getByTestId("workspace-chat-panel")).toHaveAttribute(
      "data-effective-assistant-id",
      "workspace-persona"
    )
    expect(await screen.findByText("gpt-test")).toBeInTheDocument()
    expect(screen.getByText("Analyst")).toBeInTheDocument()
    expect(
      within(
        screen.getByRole("complementary", { name: /workspace inspector/i })
      ).getByText("Streaming")
    ).toBeInTheDocument()
    expect(
      within(screen.getByLabelText("Chat workspace status")).getByText(
        "Streaming"
      )
    ).toBeInTheDocument()
  })

  it("keeps rail backend availability sourced from the connection state", async () => {
    chatPanelRuntimeState.backendAvailable = false

    render(<ChatWorkspacePage />)

    expect(
      within(
        screen.getByRole("complementary", { name: /workspace inspector/i })
      ).getByText("Streaming")
    ).toBeInTheDocument()
    expect(
      within(screen.getByLabelText("Chat workspace status")).getByText(
        "Streaming"
      )
    ).toBeInTheDocument()
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "backend:true"
    )
  })

  it("updates backend availability when the connection state changes", async () => {
    const { rerender } = render(<ChatWorkspacePage />)

    expect(
      within(screen.getByLabelText("Chat workspace status")).getByText(
        "Streaming"
      )
    ).toBeInTheDocument()

    connectionState.value = {
      phase: "error",
      isConnected: false,
      serverUrl: "http://127.0.0.1:8000"
    }
    rerender(<ChatWorkspacePage />)

    expect(
      await within(screen.getByLabelText("Chat workspace status")).findByText(
        "Server unavailable"
      )
    ).toBeInTheDocument()
  })

  it("normalizes an empty workspace id while the workspace store hydrates", () => {
    workspaceState.value = {
      workspaceId: "   ",
      workspaceName: "",
      sources: []
    }

    render(<ChatWorkspacePage />)

    const panel = screen.getByTestId("workspace-chat-panel")
    expect(panel).toHaveAttribute("data-workspace-id", "null")
    expect(panel).toHaveAttribute("data-backend-available", "false")
  })

  it("keeps chat and rails loading until the workspace store hydrates", () => {
    workspaceState.value = {
      workspaceId: "workspace-1",
      storeHydrated: false,
      workspaceName: "Default workspace",
      sources: []
    }

    const { rerender } = render(<ChatWorkspacePage />)

    expect(screen.getByTestId("workspace-chat-panel")).toHaveAttribute(
      "data-backend-available",
      "false"
    )
    expect(
      within(screen.getByLabelText("Chat workspace status")).getByText(
        "Loading workspace context"
      )
    ).toBeInTheDocument()
    expect(
      within(
        screen.getByRole("complementary", { name: /workspace inspector/i })
      ).getByText("Loading workspace context")
    ).toBeInTheDocument()

    workspaceState.value = {
      ...workspaceState.value,
      storeHydrated: true
    }
    rerender(<ChatWorkspacePage />)

    expect(screen.getByTestId("workspace-chat-panel")).toHaveAttribute(
      "data-backend-available",
      "true"
    )
    expect(
      within(screen.getByLabelText("Chat workspace status")).getByText(
        "Streaming"
      )
    ).toBeInTheDocument()
  })

  it("shows an accessible hydration failure without initializing or mounting chat", () => {
    workspaceState.value = {
      workspaceId: null,
      storeHydrated: false,
      storeHydrationError: "Stored workspace data could not be read",
      sources: []
    }

    render(<ChatWorkspacePage />)

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Stored workspace data could not be read"
    )
    expect(screen.getByRole("button", { name: "Retry workspace recovery" })).toBeEnabled()
    expect(screen.queryByTestId("workspace-chat-panel")).not.toBeInTheDocument()
    expect(workspaceActions.initializeWorkspace).not.toHaveBeenCalled()
  })

  it.each([null, undefined])(
    "activates a hydrated local workspace without a server snapshot (%j)",
    (serverWorkspace) => {
      workspaceState.value = { ...workspaceState.value, serverWorkspace }
      render(<ChatWorkspacePage />)
      const activation = screen.queryByTestId("workspace-activation")
      expect(activation).toHaveAttribute("data-workspace-id", "workspace-1")
      expect(activation).toContainElement(screen.getByTestId("chat-workspace-console"))
      expect(screen.queryByRole("link", { name: "Open workspaces" })).not.toBeInTheDocument()
      expect(workspaceActions.initializeWorkspace).not.toHaveBeenCalled()
    }
  )

  it("offers Workspaces instead of chat for a mismatched server snapshot", () => {
    workspaceState.value = {
      ...workspaceState.value,
      serverWorkspace: { metadata: { id: "different-workspace" } }
    }
    render(<ChatWorkspacePage />)
    expect(screen.getByRole("link", { name: "Open workspaces" })).toHaveAttribute("href", "/workspaces")
    expect(screen.queryByTestId("workspace-chat-panel")).not.toBeInTheDocument()
    expect(screen.queryByTestId("workspace-activation")).not.toBeInTheDocument()
  })

  it("places canonical chat behind the existing scoped activation guard", () => {
    render(<ChatWorkspacePage />)
    expect(screen.getByTestId("workspace-activation")).toHaveAttribute("data-workspace-id", "workspace-1")
    expect(screen.getByTestId("workspace-activation")).toContainElement(screen.getByTestId("workspace-chat-panel"))
  })

  it("retries persistence without treating a resolved rehydrate promise as success", async () => {
    workspaceState.value = {
      workspaceId: null,
      storeHydrated: false,
      storeHydrationError: "Stored workspace data could not be read",
      sources: []
    }
    const { rerender } = render(<ChatWorkspacePage />)

    fireEvent.click(screen.getByRole("button", { name: "Retry workspace recovery" }))
    await act(async () => { await Promise.resolve() })
    rerender(<ChatWorkspacePage />)

    expect(rehydrateWorkspace).toHaveBeenCalledTimes(1)
    expect(screen.getByRole("alert")).toHaveTextContent("could not be read")
    expect(screen.queryByTestId("workspace-chat-panel")).not.toBeInTheDocument()
    expect(workspaceActions.initializeWorkspace).not.toHaveBeenCalled()

    workspaceState.value = {
      ...workspaceState.value,
      workspaceId: "recovered-workspace",
      storeHydrated: true,
      storeHydrationError: null
    }
    rerender(<ChatWorkspacePage />)
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(screen.getByTestId("workspace-chat-panel")).toHaveAttribute(
      "data-workspace-id", "recovered-workspace"
    )
  })

  it("clears browsed and staged sources when the workspace changes", () => {
    const { rerender } = render(<ChatWorkspacePage />)

    fireEvent.click(
      screen.getByRole("button", { name: "Browse Operator Notes" })
    )
    fireEvent.click(
      screen.getByRole("button", { name: "Stage Operator Notes for chat" })
    )
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "staged:1"
    )

    workspaceState.value = {
      workspaceId: "workspace-2",
      workspaceName: "Second workspace",
      sources: [
        {
          id: "source-2",
          mediaId: 202,
          title: "Second Notes",
          type: "document",
          status: "ready",
          addedAt: new Date("2026-05-03T00:00:00Z")
        }
      ]
    }
    rerender(<ChatWorkspacePage />)

    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "staged:0"
    )
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "workspace:workspace-2"
    )
    expect(screen.queryByText("Context staged")).not.toBeInTheDocument()
  })

  it("remounts the chat panel when the workspace changes", () => {
    const { rerender } = render(<ChatWorkspacePage />)

    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "mounted:workspace-1"
    )

    workspaceState.value = {
      workspaceId: "workspace-2",
      workspaceName: "Second workspace",
      sources: [
        {
          id: "source-2",
          mediaId: 202,
          title: "Second Notes",
          type: "document",
          status: "ready",
          addedAt: new Date("2026-05-03T00:00:00Z")
        }
      ]
    }
    rerender(<ChatWorkspacePage />)

    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "mounted:workspace-2"
    )
  })

  it("ignores stale clear callbacks from a previous workspace", () => {
    const { rerender } = render(<ChatWorkspacePage />)

    fireEvent.click(
      screen.getByRole("button", { name: "Stage Operator Notes for chat" })
    )
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "staged:1"
    )
    const clearWorkspaceOne = chatPanelClearHandlers.get("workspace-1")
    expect(clearWorkspaceOne).toBeDefined()

    workspaceState.value = {
      workspaceId: "workspace-2",
      workspaceName: "Second workspace",
      sources: [
        {
          id: "source-2",
          mediaId: 202,
          title: "Second Notes",
          type: "document",
          status: "ready",
          addedAt: new Date("2026-05-03T00:00:00Z")
        }
      ]
    }
    rerender(<ChatWorkspacePage />)
    fireEvent.click(
      screen.getByRole("button", { name: "Stage Second Notes for chat" })
    )
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "staged:1"
    )

    act(() => {
      clearWorkspaceOne?.()
    })

    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "staged:1"
    )
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "workspace:workspace-2"
    )
  })

  it("ignores stale individual unstage callbacks from a previous workspace", () => {
    const { rerender } = render(<ChatWorkspacePage />)

    fireEvent.click(
      screen.getByRole("button", { name: "Stage Operator Notes for chat" })
    )
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "staged:1"
    )
    const removeWorkspaceOne = chatPanelRemoveHandlers.get("workspace-1")
    expect(removeWorkspaceOne).toBeDefined()

    workspaceState.value = {
      workspaceId: "workspace-2",
      workspaceName: "Second workspace",
      sources: [
        {
          id: "source-2",
          mediaId: 202,
          title: "Second Notes",
          type: "document",
          status: "ready",
          addedAt: new Date("2026-05-03T00:00:00Z")
        }
      ]
    }
    rerender(<ChatWorkspacePage />)
    fireEvent.click(
      screen.getByRole("button", { name: "Stage Second Notes for chat" })
    )
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "staged:1"
    )

    act(() => {
      removeWorkspaceOne?.("source-1")
    })

    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "staged:1"
    )
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "workspace:workspace-2"
    )
  })
})
