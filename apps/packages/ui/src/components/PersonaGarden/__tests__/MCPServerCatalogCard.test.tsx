import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { MCPServerCatalogCard } from "../MCPServerCatalogCard"

const baseEntry = {
  key: "github",
  name: "GitHub",
  description: "Repositories and pull requests",
  url_template: "https://api.github.com/mcp",
  auth_type: "bearer" as const,
  category: "development",
  logo_key: null,
  suggested_for: []
}

describe("MCPServerCatalogCard", () => {
  it("requires a trimmed secret before saving or testing protected servers", () => {
    const onConnect = vi.fn()
    const onTestConnection = vi.fn()

    render(
      <MCPServerCatalogCard
        entry={baseEntry}
        isRecommended={false}
        isConnected={false}
        onConnect={onConnect}
        onTestConnection={onTestConnection}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Connect" }))

    const saveButton = screen.getByRole("button", { name: "Save connection" })
    const testButton = screen.getByRole("button", { name: "Test connection" })
    const secretInput = screen.getByLabelText("Secret")

    expect(saveButton).toBeDisabled()
    expect(testButton).toBeDisabled()

    fireEvent.change(secretInput, { target: { value: "   " } })
    expect(saveButton).toBeDisabled()
    expect(testButton).toBeDisabled()

    fireEvent.change(secretInput, { target: { value: " token-123 " } })

    expect(saveButton).not.toBeDisabled()
    expect(testButton).not.toBeDisabled()

    fireEvent.click(testButton)
    fireEvent.click(saveButton)

    expect(onTestConnection).toHaveBeenCalledWith(
      "https://api.github.com/mcp",
      "bearer",
      "token-123"
    )
    expect(onConnect).toHaveBeenCalledWith({
      serverKey: "github",
      name: "GitHub",
      baseUrl: "https://api.github.com/mcp",
      authType: "bearer",
      secret: "token-123"
    })
  })

  it("clears stale secrets when authentication switches back to none", () => {
    render(
      <MCPServerCatalogCard
        entry={baseEntry}
        isRecommended={false}
        isConnected={false}
        onConnect={vi.fn()}
        onTestConnection={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Connect" }))
    fireEvent.change(screen.getByLabelText("Secret"), {
      target: { value: "secret-to-clear" }
    })
    fireEvent.change(screen.getByLabelText("Authentication type"), {
      target: { value: "none" }
    })

    expect(screen.queryByLabelText("Secret")).not.toBeInTheDocument()

    fireEvent.change(screen.getByLabelText("Authentication type"), {
      target: { value: "bearer" }
    })

    expect(screen.getByLabelText("Secret")).toHaveValue("")
  })

  it("clarifies when connection testing does not perform tool discovery", () => {
    render(
      <MCPServerCatalogCard
        entry={baseEntry}
        isRecommended={false}
        isConnected={false}
        onConnect={vi.fn()}
        onTestConnection={vi.fn()}
        testResult={{
          reachable: true,
          tools_discovered: [],
          error: null
        }}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Connect" }))

    expect(
      screen.getByText(/Tool discovery is not available for this connection test\./)
    ).toBeInTheDocument()
  })
})
