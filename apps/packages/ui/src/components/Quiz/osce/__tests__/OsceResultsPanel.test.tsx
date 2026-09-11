import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { OsceResultsPanel } from "../OsceResultsPanel"
import {
  useCompletedOsceAttemptsQuery,
  useOsceAttemptQuery
} from "../../hooks/useOsceQueries"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, value?: string | { defaultValue?: string }) =>
      typeof value === "string" ? value : value?.defaultValue ?? _key
  })
}))

vi.mock("../../hooks/useOsceQueries", () => ({
  useCompletedOsceAttemptsQuery: vi.fn(),
  useOsceAttemptQuery: vi.fn()
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const summary = {
  id: 31,
  quiz_id: 3,
  station_id: 11,
  client_attempt_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
  station_title: "Medicine safety",
  state: "completed" as const,
  version: 6,
  started_at: "2026-09-11T10:00:00Z",
  self_assessment_started_at: "2026-09-11T10:08:00Z",
  completed_at: "2026-09-11T10:12:00Z",
  last_modified_at: "2026-09-11T10:12:00Z",
  elapsed_seconds: 480,
  checklist_met_count: 2,
  checklist_total: 3,
  rubric_results: [{
    domain_id: "domain-1",
    domain_label: "Communication",
    level_id: "level-2",
    level_label: "Effective"
  }]
}

const completedAttempt = {
  id: 31,
  quiz_id: 3,
  station_id: 11,
  client_attempt_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
  state: "completed" as const,
  version: 6,
  notes: "I should pause after each warning sign.",
  started_at: "2026-09-11T10:00:00Z",
  self_assessment_started_at: "2026-09-11T10:08:00Z",
  completed_at: "2026-09-11T10:12:00Z",
  last_modified_at: "2026-09-11T10:12:00Z",
  server_time: "2026-09-11T10:12:00Z",
  elapsed_seconds: 480,
  checklist_selections: { "check-1": "met" as const },
  rubric_selections: { "domain-1": "level-2" },
  station: {
    schema_version: "osce.station.v1" as const,
    title: "Medicine safety",
    candidate_instructions: "Speak with a simulated patient.",
    candidate_task: "Explain safe medicine use.",
    patient_context: { text: "A fictional adult recently started treatment.", citations: [] },
    recommended_duration_seconds: 480,
    checklist_items: [{ id: "check-1", label: "Explain purpose", rationale: "Supports use.", citations: [] }],
    rubric_domains: [{
      id: "domain-1",
      label: "Communication",
      levels: [
        { id: "level-1", label: "Needs development", description: "Incomplete." },
        { id: "level-2", label: "Effective", description: "Clear." }
      ]
    }],
    expected_key_points: [{
      id: "point-1",
      text: "Discuss warning signs.",
      citations: [{
        source_type: "url" as const,
        source_id: "guide-1",
        label: "Safety guidance",
        source_url: "https://example.test/safety"
      }]
    }]
  }
}

describe("OsceResultsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(useCompletedOsceAttemptsQuery).mockReturnValue({
      data: {
        items: [summary],
        count: 1,
        has_more: false,
        next_offset: null,
        pagination: {
          mode: "offset",
          total: 1,
          offset: 0,
          limit: 10,
          has_more: false,
          next_offset: null
        }
      },
      isLoading: false,
      isFetching: false
    } as any)
    vi.mocked(useOsceAttemptQuery).mockReturnValue({
      data: completedAttempt,
      isLoading: false,
      isFetching: false
    } as any)
  })

  it("owns completed filters and server pagination without score language", () => {
    const { container } = render(<OsceResultsPanel />)

    expect(useCompletedOsceAttemptsQuery).toHaveBeenCalledWith(
      expect.objectContaining({ limit: 10, offset: 0 }),
      expect.any(Object)
    )
    expect(screen.getByLabelText("Filter by quiz ID")).toBeInTheDocument()
    expect(screen.getByLabelText("Filter by station ID")).toBeInTheDocument()
    expect(screen.getByText("Medicine safety")).toBeInTheDocument()
    expect(screen.getByText("2 of 3 met")).toBeInTheDocument()
    expect(screen.getByText("Communication: Effective")).toBeInTheDocument()
    expect(screen.getByText("Self-marked study practice")).toBeInTheDocument()
    expect(container.querySelector("[data-testid='osce-results-toolbar']")).toHaveClass("flex-col", "sm:flex-row")
    expect(screen.queryByText(/score|pass|fail|percentage|csv/i)).not.toBeInTheDocument()
  })

  it("updates independent filters and opens completed snapshot detail", async () => {
    render(<OsceResultsPanel />)

    fireEvent.change(screen.getByLabelText("Filter by quiz ID"), { target: { value: "3" } })
    await waitFor(() => expect(useCompletedOsceAttemptsQuery).toHaveBeenLastCalledWith(
      expect.objectContaining({ quiz_id: 3 }),
      expect.any(Object)
    ))

    fireEvent.click(screen.getByRole("button", { name: /View practice details/ }))
    await waitFor(() => expect(useOsceAttemptQuery).toHaveBeenLastCalledWith(
      31,
      expect.objectContaining({ enabled: true })
    ))
    const detailText = await screen.findByText("I should pause after each warning sign.")
    const dialog = detailText.closest("[role='dialog']")
    expect(dialog).not.toBeNull()
    if (!dialog) throw new Error("Practice detail dialog was not rendered")
    expect(dialog).toHaveAccessibleName("Medicine safety practice details")
    expect(within(dialog).getByText("I should pause after each warning sign.")).toBeInTheDocument()
    expect(within(dialog).getByText("Discuss warning signs.")).toBeInTheDocument()
    expect(within(dialog).getByRole("link", { name: "Safety guidance" })).toHaveAttribute(
      "href",
      "https://example.test/safety"
    )
    expect(within(dialog).getByText("Communication")).toBeInTheDocument()
    expect(within(dialog).getByText("Effective")).toBeInTheDocument()
  })
})
