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
    render(<WorkspaceStatusStrip backendAvailable streaming={false} stagedSourceCount={0} />)

    expect(screen.getByText("Ready via registry")).toBeInTheDocument()
    expect(screen.getByText("Ctrl+K command")).toBeInTheDocument()
    expect(screen.getByText("Ctrl+Enter send")).toBeInTheDocument()
  })

  it("renders streaming and staged context states", () => {
    render(<WorkspaceStatusStrip backendAvailable streaming stagedSourceCount={3} />)

    expect(screen.getByText("Streaming")).toBeInTheDocument()
    expect(screen.getByText("Context staged")).toBeInTheDocument()
    expect(screen.queryByText("Server unavailable")).not.toBeInTheDocument()
  })

  it("gives backend unavailable precedence over stale streaming state", () => {
    render(<WorkspaceStatusStrip backendAvailable={false} streaming stagedSourceCount={3} />)

    expect(screen.getByText("Context staged")).toBeInTheDocument()
    expect(screen.getByText("Server unavailable")).toBeInTheDocument()
    expect(screen.queryByText("Streaming")).not.toBeInTheDocument()
  })
})
