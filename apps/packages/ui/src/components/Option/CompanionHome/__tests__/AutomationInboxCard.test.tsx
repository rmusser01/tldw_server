// @vitest-environment jsdom

import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { MemoryRouter } from "react-router-dom"

import { AutomationInboxCard } from "../cards/AutomationInboxCard"

const renderCard = (
  props: Partial<React.ComponentProps<typeof AutomationInboxCard>> = {}
) =>
  render(
    <MemoryRouter>
      <AutomationInboxCard
        items={[]}
        loading={false}
        partial={false}
        error={null}
        {...props}
      />
    </MemoryRouter>
  )

describe("AutomationInboxCard", () => {
  it("renders an empty state that explains where automation results appear", () => {
    renderCard()

    expect(screen.getByRole("heading", { name: "Automation Inbox" })).toBeInTheDocument()
    expect(screen.getByText("No automation results yet")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Results and failures from scheduled tasks appear here after a run. Future scheduled questions and agent outputs appear here only when routed by task visibility policy."
      )
    ).toBeInTheDocument()
  })

  it("renders a named loading state without blocking the rest of Home", () => {
    renderCard({ loading: true })

    expect(screen.getByText("Loading automation signals")).toBeInTheDocument()
    expect(
      screen.getByText("Checking recent scheduled-task results and notifications.")
    ).toBeInTheDocument()
  })

  it("shows status, owner, summary, timestamp, and exact result links for items", () => {
    renderCard({
      items: [
        {
          id: "result:202",
          title: "Release monitor",
          summary: "Found 2 results from Release feed.",
          statusLabel: "New result",
          ownerLabel: "Watchlists",
          href: "/scheduled-tasks?tab=results&result_id=202",
          updatedAt: "2030-01-01T09:00:00Z",
          severity: "success",
          dedupeKey: "result:202"
        },
        {
          id: "run:103:state:failure",
          title: "Broken monitor",
          summary: "Broken monitor needs attention. Open details to inspect the latest run.",
          statusLabel: "Needs attention",
          ownerLabel: "Watchlists",
          href: "/scheduled-tasks?tab=results&run_id=103",
          updatedAt: "2030-01-01T09:05:00Z",
          severity: "error",
          dedupeKey: "run:103:state:failure"
        }
      ]
    })

    expect(screen.getByRole("link", { name: /Release monitor/i })).toHaveAttribute(
      "href",
      "/scheduled-tasks?tab=results&result_id=202"
    )
    expect(screen.getByText("New result")).toBeInTheDocument()
    expect(screen.getAllByText("Watchlists").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Updated Jan 1").length).toBeGreaterThan(0)
    expect(screen.getByRole("link", { name: /Broken monitor/i })).toHaveAttribute(
      "href",
      "/scheduled-tasks?tab=results&run_id=103"
    )
    expect(screen.getByText("Needs attention")).toBeInTheDocument()
  })

  it("keeps visible items while naming a partial automation load failure", () => {
    renderCard({
      partial: true,
      error: "Recent automation notifications could not be loaded.",
      items: [
        {
          id: "result:202",
          title: "Release monitor",
          summary: "Found 1 result.",
          statusLabel: "New result",
          ownerLabel: "Watchlists",
          href: "/scheduled-tasks?tab=results&result_id=202",
          updatedAt: null,
          severity: "success",
          dedupeKey: "result:202"
        }
      ]
    })

    expect(screen.getByText("Release monitor")).toBeInTheDocument()
    expect(screen.getByText("Partial automation data")).toBeInTheDocument()
    expect(
      screen.getByText("Recent automation notifications could not be loaded.")
    ).toBeInTheDocument()
  })
})
