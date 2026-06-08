// @vitest-environment jsdom

import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { ScheduledTaskCreatePanel } from "../ScheduledTaskCreatePanel"

describe("ScheduledTaskCreatePanel", () => {
  it("renders intent templates by state without source-vendor IA", () => {
    render(
      <ScheduledTaskCreatePanel
        selectedTemplateId={null}
        onSelectTemplate={vi.fn()}
        onCreateReminder={vi.fn()}
      />
    )

    expect(
      screen.getByRole("heading", { name: "Choose what you want to automate" })
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Create reminder/i })).toBeInTheDocument()
    expect(screen.getAllByText("Handoff only").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Planned capability").length).toBeGreaterThan(0)
    expect(screen.queryByRole("heading", { name: /GitHub monitor/i })).not.toBeInTheDocument()
    expect(screen.queryByRole("heading", { name: /YouTube ingest/i })).not.toBeInTheDocument()
  })

  it("suggests templates from deterministic finder text", async () => {
    const user = userEvent.setup()
    render(
      <ScheduledTaskCreatePanel
        selectedTemplateId={null}
        onSelectTemplate={vi.fn()}
        onCreateReminder={vi.fn()}
      />
    )

    await user.type(screen.getByRole("textbox", { name: "Find a template" }), "watch new advisories")

    expect(screen.getByText("Best match: Watch for new items")).toBeInTheDocument()
  })

  it("shows handoff panel copy without creation language", () => {
    render(
      <ScheduledTaskCreatePanel
        selectedTemplateId="watch"
        onSelectTemplate={vi.fn()}
        onCreateReminder={vi.fn()}
      />
    )

    expect(screen.getByText("Setup continues in Watchlists.")).toBeInTheDocument()
    expect(screen.getByText("No scheduled task has been created yet.")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Open Watchlists setup" })).toHaveAttribute(
      "href",
      "/watchlists"
    )
  })

  it("does not include sensitive URL text in handoff summary", async () => {
    const user = userEvent.setup()
    render(
      <ScheduledTaskCreatePanel
        selectedTemplateId="watch"
        onSelectTemplate={vi.fn()}
        onCreateReminder={vi.fn()}
      />
    )

    await user.type(
      screen.getByRole("textbox", { name: "Optional source or setup note" }),
      "https://example.com/feed?token=secret"
    )

    expect(screen.getByText(/private-looking values/)).toBeInTheDocument()
    expect(screen.getByLabelText("Setup summary")).not.toHaveTextContent(
      "https://example.com/feed?token=secret"
    )
  })

  it("does not include prose secrets in handoff summary", async () => {
    const user = userEvent.setup()
    render(
      <ScheduledTaskCreatePanel
        selectedTemplateId="watch"
        onSelectTemplate={vi.fn()}
        onCreateReminder={vi.fn()}
      />
    )

    await user.type(
      screen.getByRole("textbox", { name: "Optional source or setup note" }),
      "api key: sk-test-secret"
    )

    expect(screen.getByText(/private-looking values/)).toBeInTheDocument()
    expect(screen.getByLabelText("Setup summary")).not.toHaveTextContent(
      "api key: sk-test-secret"
    )
    expect(screen.getByLabelText("Setup summary")).not.toHaveTextContent("sk-test-secret")
  })

  it("renders planned Recurring question without create controls", () => {
    render(
      <ScheduledTaskCreatePanel
        selectedTemplateId="recurring_question"
        onSelectTemplate={vi.fn()}
        onCreateReminder={vi.fn()}
      />
    )

    expect(screen.getByText("Planned capability")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /Create/i })).not.toBeInTheDocument()
  })

  it("renders Reminder editor and delegates create payload", async () => {
    const user = userEvent.setup()
    const onCreateReminder = vi.fn()
    render(
      <ScheduledTaskCreatePanel
        selectedTemplateId="reminder"
        onSelectTemplate={vi.fn()}
        onCreateReminder={onCreateReminder}
      />
    )

    await user.type(screen.getByLabelText("Title"), "Review notes")
    await user.type(screen.getByLabelText("Run once at"), "2030-01-02T09:00")
    await user.click(screen.getByRole("button", { name: "Save reminder" }))

    await waitFor(() => expect(onCreateReminder).toHaveBeenCalledTimes(1))
    expect(onCreateReminder).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Review notes",
        body: null,
        schedule_kind: "one_time",
        run_at: expect.stringContaining("2030-01-02T"),
        enabled: true
      })
    )
  })
})
