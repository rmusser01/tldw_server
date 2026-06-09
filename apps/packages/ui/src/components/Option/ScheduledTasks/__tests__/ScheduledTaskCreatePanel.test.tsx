// @vitest-environment jsdom

import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { ScheduledTaskCreatePanel } from "../ScheduledTaskCreatePanel"
import {
  REQUIRED_WATCH_AVAILABILITY_GATES,
  buildScheduledTaskTemplateCapability
} from "../scheduled-task-template-capabilities"

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

  it("shows Limited availability capability copy without Watch create language", () => {
    const capability = buildScheduledTaskTemplateCapability("watch", {
      passedGates: REQUIRED_WATCH_AVAILABILITY_GATES.filter(
        (gate) => gate !== "source_preview"
      ),
      sourceIntent: {
        sourceFamily: "feed",
        can_watch: true,
        can_ingest: false,
        can_preview: false,
        can_notify: false,
        can_index_search: false,
        can_index_rag: false,
        can_create: false,
        reason: "Ingest setup continues in Watchlists."
      },
      resultDestinations: {
        home_supported: false,
        notifications_supported: false,
        search_indexed: false,
        rag_scope_included: false
      }
    })

    render(
      <ScheduledTaskCreatePanel
        selectedTemplateId="watch"
        onSelectTemplate={vi.fn()}
        onCreateReminder={vi.fn()}
        templateCapabilities={{ watch: capability }}
      />
    )

    expect(screen.getByText("Limited availability")).toBeInTheDocument()
    expect(screen.getByText(/source preview/i)).toBeInTheDocument()
    expect(screen.getByText("Detected source: feed.")).toBeInTheDocument()
    expect(screen.getByText("Watch: supported.")).toBeInTheDocument()
    expect(screen.getByText("Ingest: not supported for this source yet.")).toBeInTheDocument()
    expect(screen.getByText("Ingest setup continues in Watchlists.")).toBeInTheDocument()
    expect(screen.getByText("Home: not yet shown.")).toBeInTheDocument()
    expect(
      screen.getByText("Notifications: not available for this source yet.")
    ).toBeInTheDocument()
    expect(screen.getByText("No scheduled task has been created yet.")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /Create watch/i })).not.toBeInTheDocument()
  })

  it("keeps Available now from showing Limited availability templates", async () => {
    const user = userEvent.setup()
    render(
      <ScheduledTaskCreatePanel
        selectedTemplateId={null}
        onSelectTemplate={vi.fn()}
        onCreateReminder={vi.fn()}
        templateCapabilities={{
          watch: buildScheduledTaskTemplateCapability("watch", {
            passedGates: REQUIRED_WATCH_AVAILABILITY_GATES.filter(
              (gate) => gate !== "source_preview"
            )
          })
        }}
      />
    )

    await user.click(screen.getByText("Available now"))

    expect(screen.getByRole("button", { name: /Create reminder/i })).toBeInTheDocument()
    expect(screen.queryByText("Watch for new items")).not.toBeInTheDocument()
  })

  it("keeps capability essentials visible in an extension-width container", () => {
    const capability = buildScheduledTaskTemplateCapability("watch", {
      passedGates: REQUIRED_WATCH_AVAILABILITY_GATES.filter(
        (gate) => gate !== "source_preview"
      ),
      sourceIntent: {
        sourceFamily: "feed",
        can_watch: true,
        can_ingest: false,
        can_preview: false,
        can_notify: false,
        can_index_search: false,
        can_index_rag: false,
        can_create: false,
        reason: "Preview is not available for this source yet."
      }
    })

    render(
      <div style={{ width: 360 }}>
        <ScheduledTaskCreatePanel
          selectedTemplateId="watch"
          onSelectTemplate={vi.fn()}
          onCreateReminder={vi.fn()}
          templateCapabilities={{ watch: capability }}
        />
      </div>
    )

    expect(screen.getByText("Limited availability")).toBeInTheDocument()
    expect(screen.getByText("Detected source: feed.")).toBeInTheDocument()
    expect(screen.getByText("Preview is not available for this source yet.")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /Create watch/i })).not.toBeInTheDocument()
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
