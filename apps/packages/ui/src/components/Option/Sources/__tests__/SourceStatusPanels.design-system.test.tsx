import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { SourceStatusPanels } from "../SourceStatusPanels"
import type { IngestionSourceSummary } from "@/types/ingestion-sources"
import { getDesignSystemState } from "@/design-system"

vi.mock("@/design-system", () => ({
  getDesignSystemState: vi.fn((key: string) => ({
    key,
    label: key === "degraded" ? "Registry Degraded" : key,
    severity: key === "degraded" ? "warning" : "neutral"
  }))
}))

const makeSource = (
  overrides: Partial<IngestionSourceSummary> = {}
): IngestionSourceSummary => ({
  id: "source-1",
  user_id: 1,
  source_type: "archive_snapshot",
  sink_type: "notes",
  policy: "canonical",
  enabled: true,
  schedule_enabled: false,
  schedule_config: {},
  config: {},
  last_sync_status: "synced",
  last_successful_sync_summary: {
    changed_count: 3,
    degraded_count: 2,
    conflict_count: 0,
    sink_failure_count: 0,
    ingestion_failure_count: 0,
    created_count: 0,
    updated_count: 0,
    deleted_count: 0,
    unchanged_count: 0
  },
  ...overrides
})

describe("SourceStatusPanels design-system state labels", () => {
  beforeEach(() => {
    vi.mocked(getDesignSystemState).mockClear()
  })

  it("uses the design-system registry label for degraded summaries", () => {
    render(<SourceStatusPanels source={makeSource()} />)

    expect(screen.getByText("Registry Degraded 2")).toBeInTheDocument()
    expect(getDesignSystemState).toHaveBeenCalledWith("degraded")
  })

  it("renders degraded summaries with the design-system Badge primitive", () => {
    render(<SourceStatusPanels source={makeSource()} />)

    const badge = screen.getByText("Registry Degraded 2")

    expect(badge).toHaveAttribute("data-ds-component", "Badge")
    expect(badge).toHaveAttribute("data-ds-variant", "warning")
    expect(badge).not.toHaveClass("ant-tag")
  })
})
