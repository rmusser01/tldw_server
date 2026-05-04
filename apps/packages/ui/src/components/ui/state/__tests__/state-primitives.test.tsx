import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { PermissionNotice, RecoveryCallout, SetupRequiredPanel, StatePanel } from "../"

describe("state primitives", () => {
  it("renders canonical state labels with accessible primary actions", () => {
    render(
      <RecoveryCallout
        state="unavailable"
        title="Cannot reach the API server"
        message="Check that your server is running."
        primaryAction={{ label: "Try again", onClick: vi.fn() }}
        secondaryActions={[{ label: "Open diagnostics", onClick: vi.fn() }]}
      />
    )

    expect(screen.getByText("Unavailable")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Open diagnostics" })).toBeInTheDocument()
  })

  it("shows diagnostics only when diagnostics are provided", () => {
    render(
      <StatePanel
        state="error"
        title="Request failed"
        primaryAction={{ label: "Retry", onClick: vi.fn() }}
        diagnostics={[{ label: "Request path", value: "/api/v1/health" }]}
      />
    )

    expect(screen.getByLabelText("Diagnostics")).toBeInTheDocument()
    expect(screen.getByText("/api/v1/health")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument()
  })

  it("does not render an empty diagnostics section", () => {
    render(
      <StatePanel
        state="empty"
        title="No results"
        primaryAction={{ label: "Create source", onClick: vi.fn() }}
      />
    )

    expect(screen.queryByLabelText("Diagnostics")).not.toBeInTheDocument()
  })

  it("provides semantic wrappers for permission and setup states", () => {
    render(
      <>
        <PermissionNotice
          title="Admin access required"
          primaryAction={{ label: "Request access", onClick: vi.fn() }}
        />
        <SetupRequiredPanel
          title="Connect your server"
          primaryAction={{ label: "Open setup", onClick: vi.fn() }}
        />
      </>
    )

    expect(screen.getByText("Permission denied")).toBeInTheDocument()
    expect(screen.getByText("Setup required")).toBeInTheDocument()
  })
})
