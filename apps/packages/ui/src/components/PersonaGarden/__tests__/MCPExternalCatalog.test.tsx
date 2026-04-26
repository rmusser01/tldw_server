import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

const testConnectionMock = vi.fn()

vi.mock("@/hooks/useMCPServerCatalog", () => ({
  useMCPServerCatalog: () => ({
    entries: [
      {
        key: "slack",
        name: "Slack",
        description: "Team chat",
        url_template: "https://slack.example.com/mcp",
        auth_type: "api_key",
        category: "communication",
        logo_key: null,
        suggested_for: []
      }
    ],
    loading: false,
    error: null
  })
}))

vi.mock("@/hooks/useMCPConnectionTest", () => ({
  useMCPConnectionTest: () => ({
    test: testConnectionMock,
    result: null,
    loading: false,
    error: null
  })
}))

import { MCPExternalCatalog } from "../MCPExternalCatalog"

describe("MCPExternalCatalog", () => {
  it("forwards auth details from the card when testing a server connection", () => {
    render(
      <MCPExternalCatalog
        suggestedServers={[]}
        connectedServers={[]}
        onConnect={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Connect" }))
    fireEvent.change(screen.getByLabelText("Secret"), {
      target: { value: " slack-token " }
    })
    fireEvent.click(screen.getByRole("button", { name: "Test connection" }))

    expect(testConnectionMock).toHaveBeenCalledWith(
      "https://slack.example.com/mcp",
      "api_key",
      "slack-token"
    )
  })
})
