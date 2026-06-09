// @vitest-environment jsdom

import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { expectInsideDesignSystemAlert } from "@/test-utils/designSystemAlert"

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>()

  return {
    ...actual,
    Link: ({
      children,
      to,
      ...props
    }: React.AnchorHTMLAttributes<HTMLAnchorElement> & {
      children: React.ReactNode
      to: string
    }) => (
      <a {...props} href={to} data-router-link="true">
        {children}
      </a>
    )
  }
})

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

  it("explains Limited availability when the creation adapter is the only blocker", () => {
    const capability = buildScheduledTaskTemplateCapability("watch", {
      passedGates: REQUIRED_WATCH_AVAILABILITY_GATES,
      sourceIntent: {
        sourceFamily: "publication",
        can_watch: true,
        can_ingest: false,
        can_preview: true,
        can_notify: true,
        can_index_search: false,
        can_index_rag: false,
        can_create: false
      },
      resultDestinations: {
        home_supported: true,
        notifications_supported: true,
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
    expect(
      screen.getByText(
        "Creation from Scheduled Tasks is not available yet. Continue setup in Watchlists."
      )
    ).toBeInTheDocument()
    expect(screen.queryByText(/source preview/i)).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /Create watch/i })).not.toBeInTheDocument()
  })

  it("renders capability metadata for non-Watchlists handoff templates", () => {
    const capability = buildScheduledTaskTemplateCapability("advanced", {
      sourceIntent: {
        sourceFamily: "website",
        can_watch: false,
        can_ingest: false,
        can_preview: true,
        can_notify: false,
        can_index_search: false,
        can_index_rag: false,
        can_create: false,
        reason: "Advanced setup continues in the owner workspace."
      }
    })

    render(
      <ScheduledTaskCreatePanel
        selectedTemplateId="advanced"
        onSelectTemplate={vi.fn()}
        onCreateReminder={vi.fn()}
        templateCapabilities={{ advanced: capability }}
      />
    )

    expect(screen.getByText("Choose destination")).toBeInTheDocument()
    expect(screen.getByText("Detected source: website.")).toBeInTheDocument()
    expect(
      screen.getByText("Advanced setup continues in the owner workspace.")
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "Creation from Scheduled Tasks is not available yet. Choose the owner workspace to continue setup."
      )
    ).toBeInTheDocument()
    expect(
      screen.queryByText("Results destination: configured in Watchlists.")
    ).not.toBeInTheDocument()
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
    expectInsideDesignSystemAlert(/private-looking values/)
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
      "api key: sample credential"
    )

    expect(screen.getByText(/private-looking values/)).toBeInTheDocument()
    expect(screen.getByLabelText("Setup summary")).not.toHaveTextContent(
      "api key: sample credential"
    )
    expect(screen.getByLabelText("Setup summary")).not.toHaveTextContent("sample credential")
  })

  it("renders rich planned Recurring question guidance without create controls", () => {
    render(
      <ScheduledTaskCreatePanel
        selectedTemplateId="recurring_question"
        onSelectTemplate={vi.fn()}
        onCreateReminder={vi.fn()}
      />
    )

    expect(screen.getByText("Planned automation type")).toBeInTheDocument()
    expect(
      screen.getByText("Run this question on a schedule across selected searchable content.")
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "Recurring Question scheduling is planned for the API contract and is not executable in this client yet."
      )
    ).toBeInTheDocument()
    expect(screen.queryByText("Safety")).not.toBeInTheDocument()
    expect(screen.getByText("Scheduled RAG query support")).toBeInTheDocument()
    expect(screen.getByText("Task visibility policy")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Open Research" })).toHaveAttribute(
      "href",
      "/research"
    )
    expect(screen.getByRole("link", { name: "Open Results" })).toHaveAttribute(
      "href",
      "/scheduled-tasks/results"
    )
    expect(screen.queryByRole("button", { name: /Create/i })).not.toBeInTheDocument()
  })

  it("renders rich planned Agent Task guidance without create controls", () => {
    render(
      <ScheduledTaskCreatePanel
        selectedTemplateId="agent_task"
        onSelectTemplate={vi.fn()}
        onCreateReminder={vi.fn()}
      />
    )

    expect(screen.getByText("Planned automation type")).toBeInTheDocument()
    expect(
      screen.getByText("Send this message to the selected agent at the scheduled time.")
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "Agent Task scheduling is planned for the API contract and is not executable in this client yet."
      )
    ).toBeInTheDocument()
    expect(screen.getByText("Schedulable ACP/API agents")).toBeInTheDocument()
    expect(screen.getByText("Preview and risk classification")).toBeInTheDocument()
    expect(screen.getByText("Approval policy")).toBeInTheDocument()
    expect(
      screen.getByText("Preview is required before scheduling an agent task.")
    ).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Open Agent Tasks" })).toHaveAttribute(
      "href",
      "/agent-tasks"
    )
    expect(screen.getByRole("link", { name: "Open ACP Playground" })).toHaveAttribute(
      "href",
      "/acp-playground"
    )
    expect(screen.getByRole("link", { name: "Open Results" })).toHaveAttribute(
      "href",
      "/scheduled-tasks/results"
    )
    expect(screen.queryByRole("button", { name: /Create/i })).not.toBeInTheDocument()
  })

  it("renders planned related destinations through router links", () => {
    render(
      <ScheduledTaskCreatePanel
        selectedTemplateId="agent_task"
        onSelectTemplate={vi.fn()}
        onCreateReminder={vi.fn()}
      />
    )

    expect(screen.getByRole("link", { name: "Open Agent Tasks" })).toHaveAttribute(
      "data-router-link",
      "true"
    )
    expect(screen.getByRole("link", { name: "Open ACP Playground" })).toHaveAttribute(
      "data-router-link",
      "true"
    )
    expect(screen.getByRole("link", { name: "Open Results" })).toHaveAttribute(
      "data-router-link",
      "true"
    )
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
