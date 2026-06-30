// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { PlaygroundMcpControl } from "../PlaygroundMcpControl"

const t = (_key: string, fallback?: string) => fallback ?? _key

const renderControl = (
  overrides: Partial<React.ComponentProps<typeof PlaygroundMcpControl>> = {}
) => {
  const props: React.ComponentProps<typeof PlaygroundMcpControl> = {
    hasMcp: true,
    mcpHealthState: "healthy",
    mcpToolsLoading: false,
    mcpToolsCount: 2,
    toolChoice: "auto",
    onToolChoiceChange: vi.fn(),
    toolRunStatusLabel: "Idle",
    mcpAriaLabel: "MCP tools: Auto, 2 tools",
    mcpSummaryLabel: "2 tools",
    mcpChoiceLabel: "Auto",
    mcpDisabledReason: "",
    mcpPopoverOpen: false,
    onMcpPopoverChange: vi.fn(),
    onOpenMcpSettings: vi.fn(),
    t,
    ...overrides
  }

  render(<PlaygroundMcpControl {...props} />)

  return props
}

describe("PlaygroundMcpControl", () => {
  it("opens the MCP popover from the visible toolbar button", () => {
    const props = renderControl()

    fireEvent.click(screen.getByTestId("mcp-tools-toggle"))

    expect(props.onMcpPopoverChange).toHaveBeenCalledWith(true)
  })
})
