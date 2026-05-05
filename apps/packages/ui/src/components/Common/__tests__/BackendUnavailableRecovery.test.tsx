import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"

import BackendUnavailableRecovery from "../BackendUnavailableRecovery"

describe("BackendUnavailableRecovery", () => {
  it("renders the recovery copy, actions, and diagnostics details when provided", () => {
    render(
      <BackendUnavailableRecovery
        details={{
          title: "Can't reach your tldw server",
          message: "Check that your server is running and accessible.",
          method: "GET",
          path: "/api/v1/llm/models/metadata",
          serverUrl: "http://127.0.0.1:8000"
        }}
        onRetry={vi.fn()}
        onReload={vi.fn()}
        onOpenSettings={vi.fn()}
        onOpenDiagnostics={vi.fn()}
      />
    )

    expect(
      screen.getByRole("heading", { name: "Can't reach your tldw server" })
    ).toBeInTheDocument()
    expect(
      screen.getByText("Check that your server is running and accessible.")
    ).toBeInTheDocument()

    expect(screen.getByText("Unavailable")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Reload page" })).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Open Health & diagnostics" })
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Open Settings" })).toBeInTheDocument()

    expect(screen.getByLabelText("Diagnostics")).toBeInTheDocument()
    expect(screen.getByText("GET")).toBeInTheDocument()
    expect(screen.getByText("/api/v1/llm/models/metadata")).toBeInTheDocument()
    expect(screen.getByText("http://127.0.0.1:8000")).toBeInTheDocument()
  })

  it("keeps diagnostics hidden when no details are provided", () => {
    render(
      <BackendUnavailableRecovery
        onRetry={vi.fn()}
        onReload={vi.fn()}
        onOpenSettings={vi.fn()}
        onOpenDiagnostics={vi.fn()}
      />
    )

    expect(screen.queryByText("GET")).not.toBeInTheDocument()
    expect(
      screen.queryByText("/api/v1/llm/models/metadata")
    ).not.toBeInTheDocument()
    expect(screen.queryByText("http://127.0.0.1:8000")).not.toBeInTheDocument()
  })
})
