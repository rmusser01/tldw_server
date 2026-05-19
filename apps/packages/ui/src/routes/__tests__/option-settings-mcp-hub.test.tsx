// @vitest-environment jsdom
import React from "react"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { describe, expect, it } from "vitest"
import { render, screen } from "@testing-library/react"

import { OptionSettingsMcpHub } from "../option-settings-mcp-hub"

describe("OptionSettingsMcpHub", () => {
  it("redirects from /settings/mcp-hub to /mcp-hub", () => {
    render(
      <MemoryRouter initialEntries={["/settings/mcp-hub"]}>
        <Routes>
          <Route path="/settings/mcp-hub" element={<OptionSettingsMcpHub />} />
          <Route path="/mcp-hub" element={<div data-testid="mcp-hub-target">MCP Hub</div>} />
        </Routes>
      </MemoryRouter>
    )

    expect(screen.getByTestId("mcp-hub-target")).toBeInTheDocument()
  })
})
