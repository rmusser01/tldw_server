import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ActionGroup, PermissionNotice, RecoveryCallout, SetupRequiredPanel, StatePanel } from "../"

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

  it("keeps raw endpoint details out of the primary capability message", () => {
    render(
      <RecoveryCallout
        state="unavailable"
        title="Sources are unavailable"
        message="This server does not expose the Sources capability."
        primaryAction={{ label: "Check server setup", onClick: vi.fn() }}
        diagnostics={[
          { label: "Method", value: "GET" },
          { label: "Endpoint", value: "/api/v1/sources", code: true },
          { label: "Status", value: "404 Not Found" }
        ]}
      />
    )

    const primaryState = screen.getByRole("heading", {
      name: "Sources are unavailable"
    }).closest("div")
    const diagnostics = screen.getByLabelText("Diagnostics")

    expect(primaryState).not.toHaveTextContent("/api/v1/sources")
    expect(diagnostics).toHaveTextContent("/api/v1/sources")
    expect(screen.getByRole("button", { name: "Check server setup" })).toBeInTheDocument()
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

  it("forwards loading state to secondary actions", () => {
    render(
      <ActionGroup
        secondaryActions={[
          { label: "Sync diagnostics", onClick: vi.fn(), loading: true }
        ]}
      />
    )

    const action = screen.getByRole("button", { name: /sync diagnostics/i })
    expect(action).toBeDisabled()
    expect(action).toHaveAttribute("aria-busy", "true")
  })

  it("allows actions to keep short labels with contextual accessible names", () => {
    render(
      <ActionGroup
        secondaryActions={[
          {
            label: "Dismiss",
            ariaLabel: "Dismiss recovery suggestions",
            onClick: vi.fn()
          }
        ]}
      />
    )

    expect(screen.getByRole("button", { name: "Dismiss recovery suggestions" })).toHaveTextContent(
      "Dismiss"
    )
  })

  it("forwards live-region semantics to state panels when requested", () => {
    render(
      <RecoveryCallout
        state="degraded"
        title="Sources may need refinement"
        primaryAction={{ label: "Refine", onClick: vi.fn() }}
        role="status"
        aria-live="polite"
        aria-atomic="true"
      />
    )

    const status = screen.getByRole("status")
    expect(status).toHaveAttribute("aria-live", "polite")
    expect(status).toHaveAttribute("aria-atomic", "true")
    expect(status).toHaveAttribute("data-ds-component", "RecoveryCallout")
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
