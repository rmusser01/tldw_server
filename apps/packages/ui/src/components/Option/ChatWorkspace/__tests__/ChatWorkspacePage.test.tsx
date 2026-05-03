import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ChatWorkspacePage } from "../ChatWorkspacePage"

const setRouteContext = vi.fn()
const chatPanelRuntimeState = vi.hoisted(() => ({
  backendAvailable: true
}))

vi.mock("@/store/chat-surface-coordinator", () => ({
  useChatSurfaceCoordinatorStore: (selector: any) =>
    selector({ setRouteContext })
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (selector: any) =>
    selector({
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
    })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({
    phase: "connected",
    isConnected: true,
    serverUrl: "http://127.0.0.1:8000"
  })
}))

vi.mock("../WorkspaceChatPanel", () => ({
  WorkspaceChatPanel: ({
    stagedSources,
    workspaceId,
    onRuntimeStateChange
  }: {
    stagedSources: unknown[]
    workspaceId?: string | null
    onRuntimeStateChange?: (state: unknown) => void
  }) => {
    React.useEffect(() => {
      onRuntimeStateChange?.({
        backendAvailable: chatPanelRuntimeState.backendAvailable,
        streaming: true,
        selectedModelLabel: "gpt-test",
        selectedPersonaLabel: "Analyst"
      })
    }, [onRuntimeStateChange])

    return (
      <section data-testid="workspace-chat-panel">
        staged:{stagedSources.length}; workspace:{workspaceId}
      </section>
    )
  }
}))

describe("ChatWorkspacePage", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    chatPanelRuntimeState.backendAvailable = true
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

  it("renders backend availability from chat runtime state", async () => {
    chatPanelRuntimeState.backendAvailable = false

    render(<ChatWorkspacePage />)

    expect(
      await within(
        screen.getByRole("complementary", { name: /workspace inspector/i })
      ).findByText("Server unavailable")
    ).toBeInTheDocument()
    expect(
      within(screen.getByLabelText("Chat workspace status")).getByText(
        "Server unavailable"
      )
    ).toBeInTheDocument()
  })
})
