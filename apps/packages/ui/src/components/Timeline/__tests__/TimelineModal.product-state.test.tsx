import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { TimelineModal } from "../TimelineModal"

const timelineState = vi.hoisted(() => ({
  isOpen: true,
  isLoading: false,
  error: null as string | null,
  graph: null as { nodes: unknown[]; edges: unknown[] } | null,
  selectedNodeId: null as string | null,
  closeTimeline: vi.fn(),
  selectNode: vi.fn(),
  settings: {
    showLegend: false,
    userNodeColor: "rgb(59 130 246)",
    assistantNodeColor: "rgb(255 255 255)",
    systemNodeColor: "rgb(107 114 128)",
  },
}))

vi.mock("antd", () => ({
  Modal: ({
    children,
    onCancel,
    open,
  }: {
    children?: React.ReactNode
    onCancel?: () => void
    open?: boolean
  }) =>
    open ? (
      <div role="dialog">
        <button type="button" onClick={onCancel}>
          Close modal
        </button>
        {children}
      </div>
    ) : null,
  Spin: ({ tip }: { tip?: React.ReactNode }) => <div role="status">{tip}</div>,
}))

vi.mock("../GraphCanvas", () => ({
  GraphCanvas: () => <div data-testid="timeline-graph" />,
}))

vi.mock("../NodeInfoPanel", () => ({
  NodeInfoPanel: () => <aside data-testid="timeline-node-info" />,
}))

vi.mock("../TimelineToolbar", () => ({
  TimelineToolbar: () => <nav data-testid="timeline-toolbar" />,
}))

vi.mock("@/store/timeline", () => ({
  useTimelineStore: () => timelineState,
}))

describe("TimelineModal product states", () => {
  beforeEach(() => {
    timelineState.isOpen = true
    timelineState.isLoading = false
    timelineState.error = null
    timelineState.graph = null
    timelineState.selectedNodeId = null
    timelineState.settings.showLegend = false
    timelineState.closeTimeline.mockClear()
    timelineState.selectNode.mockClear()
  })

  afterEach(() => {
    cleanup()
  })

  it("renders loading state without timeline alerts", () => {
    timelineState.isLoading = true

    render(<TimelineModal />)

    expect(screen.getByRole("status")).toHaveTextContent(
      "Building conversation tree..."
    )
    expect(screen.queryByText("Failed to load timeline")).not.toBeInTheDocument()
    expect(screen.queryByText("No conversation data")).not.toBeInTheDocument()
  })

  it("renders the timeline error through the design-system Alert primitive", () => {
    timelineState.error = "Graph build failed"

    render(<TimelineModal />)

    const title = screen.getByText("Failed to load timeline")
    const alert = title.closest('[data-ds-component="Alert"]')
    expect(alert).toBeInTheDocument()
    expect(alert).toHaveAttribute("role", "alert")
    expect(screen.getByText("Graph build failed")).toBeInTheDocument()
  })

  it("renders the empty timeline through the design-system Alert primitive", () => {
    render(<TimelineModal />)

    const title = screen.getByText("No conversation data")
    const alert = title.closest('[data-ds-component="Alert"]')
    expect(alert).toBeInTheDocument()
    expect(alert).toHaveAttribute("role", "status")
    expect(alert).toHaveAttribute("aria-live", "polite")
    expect(
      screen.getByText("This conversation doesn't have any messages yet.")
    ).toBeInTheDocument()
  })

  it("renders the graph state and preserves close behavior", () => {
    timelineState.graph = { nodes: [], edges: [] }

    render(<TimelineModal />)

    expect(screen.getByTestId("timeline-graph")).toBeInTheDocument()
    expect(screen.queryByText("No conversation data")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Close modal" }))
    expect(timelineState.closeTimeline).toHaveBeenCalledTimes(1)
  })
})
