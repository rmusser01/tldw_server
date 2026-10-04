import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { WorkspaceStatusStrip } from "../WorkspaceStatusStrip"

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()

  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        const state = actual.getDesignSystemState(key)

        return {
          ...state,
          label: key === "ready" ? "Ready via registry" : state.label
        }
      }
    )
  }
})

describe("WorkspaceStatusStrip", () => {
  it("does not announce Ready without a model and restores readiness on selection", () => {
    const props = {
      backendAvailable: true,
      workspaceReady: true,
      streaming: false,
      stagedSourceCount: 2,
      selectedPersonaLabel: null,
      assistantSource: "none" as const
    }
    const { rerender } = render(
      <WorkspaceStatusStrip {...props} hasModelSelected={false} />
    )
    expect(screen.getByRole("status")).toHaveTextContent("Select a model")
    expect(screen.queryByText("Ready via registry")).not.toBeInTheDocument()
    expect(screen.getByRole("status")).toHaveTextContent("2 sources staged")
    expect(screen.getAllByText("Select a model")).toHaveLength(1)
    rerender(<WorkspaceStatusStrip {...props} hasModelSelected />)
    expect(screen.getByRole("status")).toHaveTextContent("Ready via registry")
    expect(screen.getByRole("status")).toHaveTextContent("No persona")
  })

  it.each([
    [{ connectionMode: "demo" }, "Demo mode - not live"],
    [{ connectionMode: "bypass" }, "Offline bypass - not verified"],
    [{ backendAvailable: false }, "Server unavailable"],
    [{ workspaceReady: false }, "Loading workspace context"],
    [{ historyLoading: true }, "Loading chat history"],
    [{ historyLoadError: "Unavailable" }, "Chat history unavailable"],
    [{ streaming: true }, "Streaming"],
    [{ sending: true }, "Sending"],
    [{ sendError: "Failed" }, "Send failed"]
  ] as const)("retains runtime precedence with no model for %j", (runtime, label) => {
    render(
      <WorkspaceStatusStrip
        backendAvailable
        workspaceReady
        streaming={false}
        hasModelSelected={false}
        stagedSourceCount={0}
        {...runtime}
      />
    )
    expect(screen.getByRole("status")).toHaveTextContent(label)
    expect(screen.queryByText("Ready via registry")).not.toBeInTheDocument()
  })

  it.each([
    [{ connectionMode: "demo" }, "Demo mode - not live"],
    [{ connectionMode: "bypass" }, "Offline bypass - not verified"],
    [{ historyLoading: true }, "Loading chat history"],
    [{ historyLoadError: "History unavailable" }, "Chat history unavailable"],
    [{ sending: true }, "Sending"]
  ] as const)("does not announce Ready for %j", (runtime, label) => {
    render(
      <WorkspaceStatusStrip
        backendAvailable
        workspaceReady
        streaming={false}
        stagedSourceCount={0}
        hasModelSelected
        {...runtime}
      />
    )
    expect(screen.getByRole("status")).toHaveTextContent(label)
    expect(screen.queryByText("Ready via registry")).not.toBeInTheDocument()
  })

  it("announces offline transitions and staged source counts", () => {
    const props = {
      backendAvailable: true,
      workspaceReady: true,
      streaming: false,
      stagedSourceCount: 0,
      hasModelSelected: true
    }
    const { rerender } = render(<WorkspaceStatusStrip {...props} />)
    expect(screen.getByRole("status")).toHaveTextContent("0 sources staged")
    rerender(
      <WorkspaceStatusStrip
        {...props}
        backendAvailable={false}
        stagedSourceCount={2}
      />
    )
    expect(screen.getByRole("status")).toHaveTextContent("Server unavailable")
    expect(screen.getByRole("status")).toHaveTextContent("2 sources staged")
  })

  it("renders ready and keyboard hint state", () => {
    render(
      <WorkspaceStatusStrip
        backendAvailable
        streaming={false}
        stagedSourceCount={0}
        workspaceReady
        hasModelSelected
        selectedPersonaLabel="Analyst"
        assistantSource="explicit"
      />
    )

    expect(screen.getByText("Ready via registry")).toBeInTheDocument()
    expect(screen.getByText("Ctrl+K command")).toBeInTheDocument()
    expect(screen.getByText("Ctrl+Enter send")).toBeInTheDocument()
  })

  it("renders streaming and staged context states", () => {
    render(
      <WorkspaceStatusStrip
        backendAvailable
        streaming
        stagedSourceCount={3}
        workspaceReady
        hasModelSelected
        selectedPersonaLabel="Analyst"
        assistantSource="explicit"
      />
    )

    expect(screen.getByText("Streaming")).toBeInTheDocument()
    expect(screen.getByText("Context staged")).toBeInTheDocument()
    expect(screen.queryByText("Server unavailable")).not.toBeInTheDocument()
  })

  it("gives backend unavailable precedence over stale streaming state", () => {
    render(
      <WorkspaceStatusStrip
        backendAvailable={false}
        streaming
        stagedSourceCount={3}
        workspaceReady
        hasModelSelected
        selectedPersonaLabel="Analyst"
        assistantSource="explicit"
      />
    )

    expect(screen.getByText("Context staged")).toBeInTheDocument()
    expect(screen.getByText("Server unavailable")).toBeInTheDocument()
    expect(screen.queryByText("Streaming")).not.toBeInTheDocument()
  })

  it("shows workspace hydration before ready when sends are disabled", () => {
    render(
      <WorkspaceStatusStrip
        backendAvailable
        streaming={false}
        stagedSourceCount={0}
        workspaceReady={false}
        hasModelSelected
        selectedPersonaLabel={null}
        assistantSource="none"
      />
    )

    expect(screen.getByText("Loading workspace context")).toBeInTheDocument()
    expect(screen.getByText("Wait for workspace identity")).toBeInTheDocument()
    expect(screen.queryByText("Ready via registry")).not.toBeInTheDocument()
  })

  it("surfaces failed sends and missing model state", () => {
    render(
      <WorkspaceStatusStrip
        backendAvailable
        streaming={false}
        stagedSourceCount={0}
        workspaceReady
        hasModelSelected={false}
        selectedPersonaLabel={null}
        assistantSource="none"
        sendError="Send failed"
      />
    )

    expect(screen.getByText("Send failed")).toBeInTheDocument()
    expect(screen.getByText("Select a model")).toBeInTheDocument()
    expect(screen.getByText("No persona")).toBeInTheDocument()
  })
})
