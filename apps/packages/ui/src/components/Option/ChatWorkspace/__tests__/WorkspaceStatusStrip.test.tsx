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
