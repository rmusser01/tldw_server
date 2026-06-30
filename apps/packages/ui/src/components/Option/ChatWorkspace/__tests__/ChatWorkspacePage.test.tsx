import React from "react"
import { act, fireEvent, render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ChatWorkspacePage } from "../ChatWorkspacePage"

const setRouteContext = vi.fn()
const chatPanelRuntimeState = vi.hoisted(() => ({
  backendAvailable: true
}))
const workspaceState = vi.hoisted(() => ({
  value: {
    workspaceId: "workspace-1",
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
    ]
  }
}))
const connectionState = vi.hoisted(() => ({
  value: {
    phase: "connected",
    isConnected: true,
    serverUrl: "http://127.0.0.1:8000"
  }
}))
const chatPanelClearHandlers = vi.hoisted(
  () => new Map<string, () => void>()
)

vi.mock("@/store/chat-surface-coordinator", () => ({
  useChatSurfaceCoordinatorStore: (selector: any) =>
    selector({ setRouteContext })
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (selector: any) =>
    selector(workspaceState.value)
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => connectionState.value
}))

vi.mock("../WorkspaceChatPanel", () => ({
  WorkspaceChatPanel: ({
    stagedSources,
    workspaceId,
    onClearStagedSources,
    backendAvailable,
    onRuntimeStateChange
  }: {
    stagedSources: unknown[]
    workspaceId?: string | null
    onClearStagedSources: () => void
    backendAvailable: boolean
    onRuntimeStateChange?: (state: unknown) => void
  }) => {
    const [mountedWorkspaceId] = React.useState(workspaceId)

    if (workspaceId) {
      chatPanelClearHandlers.set(workspaceId, onClearStagedSources)
    }

    React.useEffect(() => {
      onRuntimeStateChange?.({
        backendAvailable: chatPanelRuntimeState.backendAvailable,
        streaming: true,
        selectedModelLabel: "gpt-test",
        selectedPersonaLabel: "Analyst"
      })
    }, [onRuntimeStateChange])

    return (
      <section
        data-testid="workspace-chat-panel"
        data-workspace-id={workspaceId ?? "null"}
        data-backend-available={String(backendAvailable)}
      >
        staged:{stagedSources.length}; workspace:{workspaceId}; mounted:
        {mountedWorkspaceId}; backend:{String(backendAvailable)}
      </section>
    )
  }
}))

describe("ChatWorkspacePage", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    chatPanelRuntimeState.backendAvailable = true
    workspaceState.value = {
      workspaceId: "workspace-1",
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
      ]
    }
    connectionState.value = {
      phase: "connected",
      isConnected: true,
      serverUrl: "http://127.0.0.1:8000"
    }
    chatPanelClearHandlers.clear()
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

    fireEvent.click(screen.getByRole("button", { name: "Browse Operator Notes" }))
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent("staged:0")

    fireEvent.click(
      screen.getByRole("button", { name: "Stage Operator Notes for chat" })
    )
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent("staged:1")
  })

  it("passes workspace scope and real runtime state into the visible rails", async () => {
    render(<ChatWorkspacePage />)

    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "workspace:workspace-1"
    )
    expect(await screen.findByText("gpt-test")).toBeInTheDocument()
    expect(screen.getByText("Analyst")).toBeInTheDocument()
    expect(
      within(
        screen.getByRole("complementary", { name: /workspace inspector/i })
      ).getByText("Streaming")
    ).toBeInTheDocument()
    expect(
      within(screen.getByLabelText("Chat workspace status")).getByText("Streaming")
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
      within(screen.getByLabelText("Chat workspace status")).getByText("Streaming")
    ).toBeInTheDocument()
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "backend:true"
    )
  })

  it("updates backend availability when the connection state changes", async () => {
    const { rerender } = render(<ChatWorkspacePage />)

    expect(
      within(screen.getByLabelText("Chat workspace status")).getByText("Streaming")
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

  it("clears browsed and staged sources when the workspace changes", () => {
    const { rerender } = render(<ChatWorkspacePage />)

    fireEvent.click(screen.getByRole("button", { name: "Browse Operator Notes" }))
    fireEvent.click(
      screen.getByRole("button", { name: "Stage Operator Notes for chat" })
    )
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent("staged:1")

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
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent("staged:1")
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
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent("staged:1")

    act(() => {
      clearWorkspaceOne?.()
    })

    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent("staged:1")
    expect(screen.getByTestId("workspace-chat-panel")).toHaveTextContent(
      "workspace:workspace-2"
    )
  })
})
