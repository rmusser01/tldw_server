// @vitest-environment jsdom

import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  useCanonicalConnectionConfig: vi.fn(),
  listScheduledTasks: vi.fn(),
  createScheduledTaskReminder: vi.fn(),
  updateScheduledTaskReminder: vi.fn(),
  deleteScheduledTaskReminder: vi.fn()
}))

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: (...args: unknown[]) =>
    mocks.useCanonicalConnectionConfig(...args)
}))

vi.mock("@/services/scheduled-tasks-control-plane", () => ({
  listScheduledTasks: (...args: unknown[]) => mocks.listScheduledTasks(...args),
  createScheduledTaskReminder: (...args: unknown[]) => mocks.createScheduledTaskReminder(...args),
  updateScheduledTaskReminder: (...args: unknown[]) => mocks.updateScheduledTaskReminder(...args),
  deleteScheduledTaskReminder: (...args: unknown[]) => mocks.deleteScheduledTaskReminder(...args)
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) => {
      if (typeof fallback === "string") {
        return fallback
      }
      return fallback?.defaultValue ?? _key
    }
  })
}))

import { ScheduledTasksPage } from "../ScheduledTasksPage"

const fetchMock = vi.fn()
vi.stubGlobal("fetch", fetchMock)

const SLOW_SCHEDULE_FORM_TIMEOUT_MS = 20000

const renderWithQueryClient = (
  ui: React.ReactElement,
  initialEntry = "/scheduled-tasks"
) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })

  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>
    </MemoryRouter>
  )
}

describe("ScheduledTasksPage", () => {
  beforeEach(() => {
    for (const mock of Object.values(mocks)) {
      mock.mockReset()
    }
    mocks.useCanonicalConnectionConfig.mockReturnValue({
      config: {
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "test-key"
      },
      loading: false
    })
    fetchMock.mockReset()
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        paths: {
          "/api/v1/scheduled-tasks": {}
        }
      })
    })
  })

  it("shows an unsupported-state message without calling the list endpoint when scheduled tasks are unavailable", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        paths: {}
      })
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    expect(await screen.findByText("Unavailable")).toBeInTheDocument()
    expect(
      await screen.findByRole("heading", {
        name: "Scheduled tasks are unavailable on this server"
      })
    ).toBeInTheDocument()
    expect(
      screen.getByText("The connected server does not advertise scheduled task management.")
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Diagnostics")).toHaveTextContent("/api/v1/scheduled-tasks")
    expect(screen.getByRole("button", { name: "Health & diagnostics" })).toBeInTheDocument()
    expect(mocks.listScheduledTasks).not.toHaveBeenCalled()
  })

  it("passes an abort signal to the scheduled-tasks support probe", async () => {
    let receivedSignal: AbortSignal | undefined
    fetchMock.mockImplementationOnce((_url, init?: RequestInit) => {
      receivedSignal = init?.signal ?? undefined
      return Promise.resolve({
        ok: true,
        json: async () => ({
          paths: {
            "/api/v1/scheduled-tasks": {}
          }
        })
      })
    })
    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    expect(await screen.findByText("No scheduled tasks yet.")).toBeInTheDocument()
    expect(receivedSignal).toBeInstanceOf(AbortSignal)
  })

  it("shows auth-required recovery copy for scheduled task load failures", async () => {
    mocks.listScheduledTasks.mockRejectedValue(
      Object.assign(new Error("Request failed: 401 (GET /api/v1/scheduled-tasks)"), {
        status: 401
      })
    )

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    expect(
      await screen.findByRole("heading", { name: "Sign in before using scheduled tasks" })
    ).toBeInTheDocument()
    expect(
      screen.getByText("Connect or repair your tldw credentials, then try again.")
    ).toBeInTheDocument()

    const diagnostics = screen.getByLabelText("Diagnostics")
    expect(within(diagnostics).getByText("/api/v1/scheduled-tasks")).toBeInTheDocument()
    expect(within(diagnostics).getByText("401")).toBeInTheDocument()
    expect(
      within(diagnostics).getByText("Request failed: 401 (GET /api/v1/scheduled-tasks)")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Health & diagnostics" })).toBeInTheDocument()
  })

  it("keeps loaded rows visible when one scheduled task dependency fails", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [
        {
          id: "reminder_task:partial",
          primitive: "reminder_task",
          title: "Loaded reminder",
          description: "This row still rendered",
          status: "scheduled",
          enabled: true,
          schedule_summary: "Every weekday",
          timezone: "UTC",
          next_run_at: "2030-01-02T09:00:00+00:00",
          last_run_at: null,
          edit_mode: "native",
          manage_url: null,
          source_ref: { task_id: "partial" }
        }
      ],
      total: 1,
      partial: true,
      errors: ["Watchlist jobs failed at /api/v1/watchlists/jobs"]
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    expect(
      await screen.findByRole("heading", {
        name: "Scheduled tasks are partially available"
      })
    ).toBeInTheDocument()
    expect(
      screen.getByText("Some scheduled-task data loaded while one dependency could not be reached.")
    ).toBeInTheDocument()
    expect(await screen.findByText("Loaded reminder")).toBeInTheDocument()

    const diagnostics = screen.getByLabelText("Diagnostics")
    expect(within(diagnostics).getByText("Watchlist jobs failed at /api/v1/watchlists/jobs")).toBeInTheDocument()
  })

  it("opens the Create tab from the URL", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=create")

    expect(
      await screen.findByRole("heading", {
        level: 3,
        name: "Choose what you want to automate"
      })
    ).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Create" })).toHaveAttribute("aria-selected", "true")
  })

  it("opens the Results tab from the URL and renders projected result signals", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [
        {
          id: "watchlist_job:release",
          primitive: "watchlist_job",
          title: "Release monitor",
          description: "Track releases",
          status: "scheduled",
          enabled: true,
          schedule_summary: "Every morning",
          timezone: "UTC",
          next_run_at: "2030-04-06T09:00:00+00:00",
          last_run_at: "2030-04-05T09:00:00+00:00",
          edit_mode: "external",
          manage_url: "/watchlists?tab=jobs",
          source_ref: {
            job_id: 42,
            latest_run_id: 101,
            latest_output_id: 202,
            result_count: 3,
            source_label: "Release feed"
          }
        }
      ],
      total: 1,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=results")

    expect(await screen.findByRole("tab", { name: "Results" })).toHaveAttribute("aria-selected", "true")
    expect(await screen.findByRole("heading", { level: 3, name: "Scheduled task results" })).toBeInTheDocument()
    expect(screen.getByText("Latest signals inferred from task status. Durable review state appears when the results API is available.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Open signal for Release monitor" })).toBeInTheDocument()
    expect(screen.getByText("Found 3 results from Release feed.")).toBeInTheDocument()
  })

  it("opens the Results tab from the alias path and selects a result signal", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [
        {
          id: "watchlist_job:release",
          primitive: "watchlist_job",
          title: "Release monitor",
          description: "Track releases",
          status: "scheduled",
          enabled: true,
          schedule_summary: "Every morning",
          timezone: "UTC",
          next_run_at: "2030-04-06T09:00:00+00:00",
          last_run_at: "2030-04-05T09:00:00+00:00",
          edit_mode: "external",
          manage_url: "/watchlists?tab=jobs",
          source_ref: {
            job_id: 42,
            latest_run_id: 101,
            latest_output_id: 202,
            result_count: 1
          }
        }
      ],
      total: 1,
      partial: false,
      errors: []
    })

    renderWithQueryClient(
      <ScheduledTasksPage />,
      "/scheduled-tasks/results?result_id=202"
    )

    expect(await screen.findByRole("tab", { name: "Results" })).toHaveAttribute("aria-selected", "true")
    expect(await screen.findByText("Selected signal: Release monitor")).toBeInTheDocument()
    expect(screen.queryByText("Result signal not found.")).not.toBeInTheDocument()
  })

  it("shows a non-blocking missing-result message for stale Results deep links", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(
      <ScheduledTasksPage />,
      "/scheduled-tasks?tab=results&result_id=missing"
    )

    expect(await screen.findByRole("tab", { name: "Results" })).toHaveAttribute("aria-selected", "true")
    expect(await screen.findByText("Result signal not found.")).toBeInTheDocument()
    expect(screen.getByText("No scheduled tasks yet")).toBeInTheDocument()
  })

  it("keeps Watch template non-creating from the page route", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=create&template=watch")

    expect(await screen.findByText("No scheduled task has been created yet.")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /Create watch/i })).not.toBeInTheDocument()
  })

  it("opens a task detail deep link after task data loads", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [
        {
          id: "reminder_task:1",
          primitive: "reminder_task",
          title: "Review notes",
          description: "Check the backlog",
          status: "scheduled",
          enabled: true,
          schedule_summary: "2026-03-21T09:00:00+00:00",
          timezone: "UTC",
          next_run_at: "2030-04-05T12:30:00+00:00",
          last_run_at: null,
          edit_mode: "native",
          manage_url: null,
          source_ref: { task_id: "1" }
        }
      ],
      total: 1,
      partial: false,
      errors: []
    })

    renderWithQueryClient(
      <ScheduledTasksPage />,
      "/scheduled-tasks?tab=tasks&task_id=reminder_task%3A1"
    )

    expect(await screen.findByRole("tab", { name: "Tasks" })).toHaveAttribute("aria-selected", "true")
    const drawer = await screen.findByRole("dialog", { name: /Review notes/i })
    expect(within(drawer).getByText("Reminder")).toBeInTheDocument()
  })

  it("falls back to Overview with non-blocking copy for an invalid tab", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=runs")

    expect(await screen.findByText("That tab is not available. Showing Overview.")).toBeInTheDocument()
    expect(await screen.findByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true")
    expect(await screen.findByText("0 scheduled tasks")).toBeInTheDocument()
  })

  it("keeps the Tasks tab visible for a missing task deep link", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(
      <ScheduledTasksPage />,
      "/scheduled-tasks?tab=tasks&task_id=reminder_task%3Amissing"
    )

    expect(await screen.findByText("Task not found.")).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Tasks" })).toHaveAttribute("aria-selected", "true")
    expect(await screen.findByText("No scheduled tasks yet.")).toBeInTheDocument()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("renders the workbench overview and Watchlists preservation copy", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [
        {
          id: "reminder_task:1",
          primitive: "reminder_task",
          title: "Review notes",
          description: "Check the backlog",
          status: "failed with results",
          enabled: true,
          schedule_summary: "2026-03-21T09:00:00+00:00",
          timezone: "UTC",
          next_run_at: "2030-04-05T12:30:00+00:00",
          last_run_at: null,
          edit_mode: "native",
          manage_url: null,
          source_ref: { task_id: "1" }
        },
        {
          id: "watchlist_job:2",
          primitive: "watchlist_job",
          title: "Morning digest",
          description: "Watchlist run",
          status: "running",
          enabled: true,
          schedule_summary: "0 9 * * *",
          timezone: "UTC",
          next_run_at: "2030-04-06T09:00:00+00:00",
          last_run_at: "2030-04-05T09:00:00+00:00",
          edit_mode: "external",
          manage_url: "/watchlists?tab=jobs",
          source_ref: { job_id: 2, latest_run_id: 25, latest_output_id: 39 }
        }
      ],
      total: 2,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />)

    expect(await screen.findByRole("heading", { level: 2, name: "Scheduled tasks" })).toBeInTheDocument()
    expect(
      screen.getByText(
        "Track reminders, Watchlist monitors, and recurring automation from one place. Use domain workspaces like Watchlists for deep source and output configuration."
      )
    ).toBeInTheDocument()
    expect(await screen.findByText("2 scheduled tasks")).toBeInTheDocument()
    expect(screen.getByText("1 needs attention")).toBeInTheDocument()
    expect(screen.getByText("1 running now")).toBeInTheDocument()
    expect(screen.getByText("Next upcoming run")).toBeInTheDocument()
    expect(screen.getAllByText(/2030/).length).toBeGreaterThan(0)
    expect(screen.getByText(/Watchlists remains the full workspace/)).toBeInTheDocument()
    expect(screen.queryByText("Review notes")).not.toBeInTheDocument()
  })

  it("renders task rows and Watchlists links inside the Tasks tab", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [
        {
          id: "reminder_task:1",
          primitive: "reminder_task",
          title: "Review notes",
          description: "Check the backlog",
          status: "failed with results",
          enabled: true,
          schedule_summary: "2026-03-21T09:00:00+00:00",
          timezone: "UTC",
          next_run_at: "2030-04-05T12:30:00+00:00",
          last_run_at: null,
          edit_mode: "native",
          manage_url: null,
          source_ref: { task_id: "1" }
        },
        {
          id: "watchlist_job:2",
          primitive: "watchlist_job",
          title: "Morning digest",
          description: "Watchlist run",
          status: "running",
          enabled: true,
          schedule_summary: "0 9 * * *",
          timezone: "UTC",
          next_run_at: "2030-04-06T09:00:00+00:00",
          last_run_at: "2030-04-05T09:00:00+00:00",
          edit_mode: "external",
          manage_url: "/watchlists?tab=jobs",
          source_ref: { job_id: 2, latest_run_id: 25, latest_output_id: 39 }
        }
      ],
      total: 2,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    expect(await screen.findByText("Review notes")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create scheduled task" })).toBeInTheDocument()
    expect(screen.getAllByText("Reminder").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Watchlist monitor").length).toBeGreaterThan(0)
    expect(screen.getByText("Managed here")).toBeInTheDocument()
    expect(screen.getByText("Managed in Watchlists")).toBeInTheDocument()
    expect(screen.getByRole("columnheader", { name: "Last run" })).toBeInTheDocument()
    expect(screen.getByRole("columnheader", { name: "Next run" })).toBeInTheDocument()

    const reminderRow = screen.getByText("Review notes").closest("tr")
    expect(reminderRow).not.toBeNull()
    expect(within(reminderRow as HTMLElement).getByText("Needs attention")).toBeInTheDocument()
    expect(within(reminderRow as HTMLElement).getByText("No completed runs yet")).toBeInTheDocument()

    expect(screen.getByRole("button", { name: "Inspect Review notes" })).toBeInTheDocument()
    expect(await screen.findByRole("button", { name: "Edit Review notes" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Delete Review notes" })).toBeInTheDocument()
    expect(await screen.findByText("Morning digest")).toBeInTheDocument()
    expect(
      await screen.findByRole("link", { name: "Open monitor settings for Morning digest" })
    ).toHaveAttribute("href", "/watchlists?tab=jobs")
    expect(screen.getByRole("link", { name: "Open activity for Morning digest" })).toHaveAttribute(
      "href",
      "/watchlists?tab=runs&job_id=2"
    )
    expect(screen.getByRole("link", { name: "Open reports for Morning digest" })).toHaveAttribute(
      "href",
      "/watchlists?tab=outputs&job_id=2"
    )
    expect(screen.getByRole("link", { name: "Open latest run for Morning digest" })).toHaveAttribute(
      "href",
      "/watchlists?tab=runs&run_id=25&open_run=1"
    )
    expect(screen.getByRole("link", { name: "Open latest report for Morning digest" })).toHaveAttribute(
      "href",
      "/watchlists?tab=outputs&output_id=39&open_output=1"
    )
    expect(screen.queryByRole("button", { name: "Edit watchlist job" })).not.toBeInTheDocument()
  })

  it("opens the detail drawer for the inspected scheduled task row", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks.mockResolvedValue({
      items: [
        {
          id: "reminder_task:1",
          primitive: "reminder_task",
          title: "Review notes",
          description: "Check the backlog",
          status: "scheduled",
          enabled: true,
          schedule_summary: "2026-03-21T09:00:00+00:00",
          timezone: "UTC",
          next_run_at: "2030-04-05T12:30:00+00:00",
          last_run_at: null,
          edit_mode: "native",
          manage_url: null,
          source_ref: { task_id: "1" }
        }
      ],
      total: 1,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    await user.click(await screen.findByRole("button", { name: "Inspect Review notes" }))

    const drawer = await screen.findByRole("dialog", { name: /Review notes/i })
    expect(within(drawer).getByText("Reminder")).toBeInTheDocument()
    expect(within(drawer).getByRole("button", { name: "Edit reminder" })).toBeInTheDocument()
  })

  it("closes the detail drawer when a refetch removes the inspected task", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks
      .mockResolvedValueOnce({
        items: [
          {
            id: "reminder_task:stale",
            primitive: "reminder_task",
            title: "Stale reminder",
            description: "Will be removed",
            status: "scheduled",
            enabled: true,
            schedule_summary: "Every weekday",
            timezone: "UTC",
            next_run_at: "2030-04-05T12:30:00+00:00",
            last_run_at: null,
            edit_mode: "native",
            manage_url: null,
            source_ref: { task_id: "stale" }
          }
        ],
        total: 1,
        partial: true,
        errors: ["Watchlists jobs temporarily unavailable"]
      })
      .mockResolvedValueOnce({
        items: [],
        total: 0,
        partial: false,
        errors: []
      })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    await user.click(await screen.findByRole("button", { name: "Inspect Stale reminder" }))
    expect(await screen.findByRole("dialog", { name: /Stale reminder/i })).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Try again" }))

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /Stale reminder/i })).not.toBeInTheDocument()
    })
    expect(await screen.findByText("No scheduled tasks yet.")).toBeInTheDocument()
  })

  it("opens the reminder editor from the detail drawer without leaving the drawer open", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks.mockResolvedValue({
      items: [
        {
          id: "reminder_task:1",
          primitive: "reminder_task",
          title: "Review notes",
          description: "Check the backlog",
          status: "scheduled",
          enabled: true,
          schedule_summary: "2026-03-21T09:00:00+00:00",
          timezone: "UTC",
          next_run_at: "2030-04-05T12:30:00+00:00",
          last_run_at: null,
          edit_mode: "native",
          manage_url: null,
          source_ref: { task_id: "1", schedule_kind: "one_time", run_at: "2026-03-21T09:00:00+00:00" }
        }
      ],
      total: 1,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    await user.click(await screen.findByRole("button", { name: "Inspect Review notes" }))
    expect(await screen.findByRole("dialog", { name: /Review notes/i })).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Edit reminder" }))

    expect(await screen.findByText("Edit reminder")).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /Review notes/i })).not.toBeInTheDocument()
    })
  })

  it("deletes the selected reminder from the detail drawer and does not leave stale drawer state", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks
      .mockResolvedValueOnce({
        items: [
          {
            id: "reminder_task:1",
            primitive: "reminder_task",
            title: "Review notes",
            description: "Check the backlog",
            status: "scheduled",
            enabled: true,
            schedule_summary: "2026-03-21T09:00:00+00:00",
            timezone: "UTC",
            next_run_at: "2030-04-05T12:30:00+00:00",
            last_run_at: null,
            edit_mode: "native",
            manage_url: null,
            source_ref: { task_id: "1" }
          }
        ],
        total: 1,
        partial: false,
        errors: []
      })
      .mockResolvedValueOnce({
        items: [],
        total: 0,
        partial: false,
        errors: []
      })
    mocks.deleteScheduledTaskReminder.mockResolvedValue({ deleted: true })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    await user.click(await screen.findByRole("button", { name: "Inspect Review notes" }))
    expect(await screen.findByRole("dialog", { name: /Review notes/i })).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Delete reminder" }))

    await waitFor(() => {
      expect(mocks.deleteScheduledTaskReminder).toHaveBeenCalledWith("reminder_task:1")
    })
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /Review notes/i })).not.toBeInTheDocument()
    })
    expect(await screen.findByText("No scheduled tasks yet.")).toBeInTheDocument()
  })

  it("filters scheduled tasks by product status and search text", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks.mockResolvedValue({
      items: [
        {
          id: "reminder_task:healthy",
          primitive: "reminder_task",
          title: "Healthy reminder",
          description: "Runs normally",
          status: "scheduled",
          enabled: true,
          schedule_summary: "Every weekday",
          timezone: "UTC",
          next_run_at: "2030-04-05T12:30:00+00:00",
          last_run_at: null,
          edit_mode: "native",
          manage_url: null,
          source_ref: { task_id: "healthy" }
        },
        {
          id: "watchlist_job:blocked",
          primitive: "watchlist_job",
          title: "Blocked monitor",
          description: "Needs credentials",
          status: "blocked",
          enabled: true,
          schedule_summary: "Every morning",
          timezone: "UTC",
          next_run_at: null,
          last_run_at: null,
          edit_mode: "external",
          manage_url: "/watchlists?tab=jobs",
          source_ref: { job_id: 42 }
        }
      ],
      total: 2,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    expect(await screen.findByText("Healthy reminder")).toBeInTheDocument()
    expect(screen.getByText("Blocked monitor")).toBeInTheDocument()
    const healthyRow = screen.getByText("Healthy reminder").closest("tr")
    expect(healthyRow).not.toBeNull()
    expect(within(healthyRow as HTMLElement).getByText("Waiting for next run")).toBeInTheDocument()
    expect(within(healthyRow as HTMLElement).queryByText("scheduled")).not.toBeInTheDocument()

    await user.click(screen.getByRole("combobox", { name: "Status filter" }))
    await user.click(await screen.findByTitle("Needs attention"))

    expect(screen.queryByText("Healthy reminder")).not.toBeInTheDocument()
    expect(screen.getByText("Blocked monitor")).toBeInTheDocument()

    await user.click(screen.getByRole("combobox", { name: "Status filter" }))
    await user.click(await screen.findByTitle("All statuses"))
    fireEvent.change(screen.getByRole("textbox", { name: "Search scheduled tasks" }), {
      target: { value: "healthy" }
    })

    expect(screen.getByText("Healthy reminder")).toBeInTheDocument()
    expect(screen.queryByText("Blocked monitor")).not.toBeInTheDocument()
  })

  it("counts blocked tasks as needing attention in the overview", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [
        {
          id: "watchlist_job:blocked",
          primitive: "watchlist_job",
          title: "Blocked monitor",
          description: "Needs credentials",
          status: "blocked",
          enabled: true,
          schedule_summary: "Every morning",
          timezone: "UTC",
          next_run_at: "2030-04-06T09:00:00+00:00",
          last_run_at: null,
          edit_mode: "external",
          manage_url: "/watchlists?tab=jobs",
          source_ref: { job_id: 42 }
        }
      ],
      total: 1,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />)

    expect(await screen.findByText("1 needs attention")).toBeInTheDocument()
  })

  it("ignores disabled tasks when choosing the next upcoming run", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [
        {
          id: "reminder_task:disabled",
          primitive: "reminder_task",
          title: "Disabled stale reminder",
          description: "Old disabled run",
          status: "scheduled",
          enabled: false,
          schedule_summary: "Disabled one-time reminder",
          timezone: "UTC",
          next_run_at: "2029-01-01T09:00:00+00:00",
          last_run_at: null,
          edit_mode: "native",
          manage_url: null,
          source_ref: { task_id: "disabled" }
        },
        {
          id: "reminder_task:enabled",
          primitive: "reminder_task",
          title: "Enabled reminder",
          description: "Upcoming enabled run",
          status: "scheduled",
          enabled: true,
          schedule_summary: "Enabled one-time reminder",
          timezone: "UTC",
          next_run_at: "2030-05-06T09:00:00+00:00",
          last_run_at: null,
          edit_mode: "native",
          manage_url: null,
          source_ref: { task_id: "enabled" }
        }
      ],
      total: 2,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />)

    const overview = await screen.findByLabelText("Scheduled task overview")
    expect(within(overview).getByText(/2030/)).toBeInTheDocument()
    expect(within(overview).queryByText(/2029/)).not.toBeInTheDocument()
  })

  it("shows a clear loading state while scheduled task data loads", async () => {
    mocks.listScheduledTasks.mockReturnValue(new Promise(() => undefined))

    renderWithQueryClient(<ScheduledTasksPage />)

    expect(await screen.findByText("Loading tasks and latest run state")).toBeInTheDocument()
  })

  it("shows an actionable empty state when no scheduled tasks exist", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    expect(await screen.findByText("No scheduled tasks yet.")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Create scheduled task" }))
    expect(
      await screen.findByRole("heading", {
        level: 3,
        name: "Choose what you want to automate"
      })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("heading", { level: 4, name: "Scheduled tasks" })
    ).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Save reminder" })).not.toBeInTheDocument()
    expect(
      screen.getByText(
        "Create a reminder now. Watch and Ingest setup continue in their owner workspaces until capability, preview, duplicate, creation, and result contracts are available."
      )
    ).toBeInTheDocument()
    expect(screen.queryByText(/GitHub, YouTube/i)).not.toBeInTheDocument()
  })

  it("opens the created reminder detail after successful creation", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks
      .mockResolvedValueOnce({
        items: [],
        total: 0,
        partial: false,
        errors: []
      })
      .mockResolvedValueOnce({
        items: [
          {
            id: "reminder_task:2",
            primitive: "reminder_task",
            title: "Daily review",
            description: null,
            status: "scheduled",
            enabled: true,
            schedule_summary: "2026-03-21T10:00:00+00:00",
            timezone: "UTC",
            next_run_at: "2026-03-21T10:00:00+00:00",
            last_run_at: null,
            edit_mode: "native",
            manage_url: null,
            source_ref: { task_id: "2" }
          }
        ],
        total: 1,
        partial: false,
        errors: []
      })
    mocks.createScheduledTaskReminder.mockResolvedValue({
      id: "reminder_task:2",
      primitive: "reminder_task",
      title: "Daily review",
      description: null,
      status: "scheduled",
      enabled: true,
      edit_mode: "native",
      source_ref: { task_id: "2" }
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=create&template=reminder")

    await user.type(await screen.findByRole("textbox", { name: "Title" }), "Daily review")
    fireEvent.change(screen.getByLabelText("Run once at"), {
      target: { value: "2026-03-21T10:00" }
    })
    await user.click(await screen.findByRole("button", { name: "Save reminder" }))

    await waitFor(() => {
      expect(mocks.createScheduledTaskReminder).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Daily review",
          schedule_kind: "one_time",
          run_at: expect.stringMatching(/^2026-03-21T\d{2}:00:00\.000Z$/),
          enabled: true
        })
      )
    })
    expect(await screen.findByRole("tab", { name: "Tasks" })).toHaveAttribute("aria-selected", "true")
    const drawer = await screen.findByRole("dialog", { name: /Daily review/i })
    expect(within(drawer).getByText("Reminder")).toBeInTheDocument()
    expect(mocks.listScheduledTasks).toHaveBeenCalledTimes(2)
  })

  it("keeps the created reminder detail open from the API response until the list catches up", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks
      .mockResolvedValueOnce({
        items: [],
        total: 0,
        partial: false,
        errors: []
      })
      .mockResolvedValueOnce({
        items: [],
        total: 0,
        partial: false,
        errors: []
      })
    mocks.createScheduledTaskReminder.mockResolvedValue({
      id: "reminder_task:pending",
      primitive: "reminder_task",
      title: "Pending reminder",
      description: null,
      status: "scheduled",
      enabled: true,
      schedule_summary: "2026-03-21T10:00:00+00:00",
      timezone: "UTC",
      next_run_at: "2026-03-21T10:00:00+00:00",
      last_run_at: null,
      edit_mode: "native",
      manage_url: null,
      source_ref: { task_id: "pending" }
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=create&template=reminder")

    await user.type(await screen.findByRole("textbox", { name: "Title" }), "Pending reminder")
    fireEvent.change(screen.getByLabelText("Run once at"), {
      target: { value: "2026-03-21T10:00" }
    })
    await user.click(await screen.findByRole("button", { name: "Save reminder" }))

    const drawer = await screen.findByRole("dialog", { name: /Pending reminder/i })
    expect(within(drawer).getByText("Reminder")).toBeInTheDocument()
    expect(screen.queryByText("Task not found.")).not.toBeInTheDocument()
  })

  it("does not create a one-time reminder without run_at", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=create&template=reminder")

    await user.type(await screen.findByRole("textbox", { name: "Title" }), "Missing run at")
    await user.click(await screen.findByRole("button", { name: "Save reminder" }))

    await waitFor(() => {
      expect(mocks.createScheduledTaskReminder).not.toHaveBeenCalled()
    })
    expect(screen.getByText("Run at is required for one-time reminders")).toBeInTheDocument()
  })

  it("creates a daily recurring reminder with cron and timezone from safer controls", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })
    mocks.createScheduledTaskReminder.mockResolvedValue({
      id: "reminder_task:daily",
      primitive: "reminder_task",
      title: "Daily recurring review",
      description: null,
      status: "scheduled",
      enabled: true,
      edit_mode: "native",
      source_ref: { task_id: "daily" }
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=create&template=reminder")

    await user.type(await screen.findByRole("textbox", { name: "Title" }), "Daily recurring review")
    fireEvent.click(screen.getByText("Repeat"))
    expect(await screen.findByRole("combobox", { name: "Repeat preset" })).toBeInTheDocument()
    fireEvent.change(await screen.findByLabelText("Timezone"), {
      target: { value: "America/Los_Angeles" }
    })
    await user.click(await screen.findByRole("button", { name: "Save reminder" }))

    await waitFor(() => {
      expect(mocks.createScheduledTaskReminder).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Daily recurring review",
          schedule_kind: "recurring",
          cron: "0 9 * * *",
          timezone: "America/Los_Angeles",
          enabled: true
        })
      )
    })
  })

  it("does not create a recurring reminder without cron and timezone", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=create&template=reminder")

    await user.type(await screen.findByRole("textbox", { name: "Title" }), "Recurring reminder")
    fireEvent.click(await screen.findByText("Repeat"))
    await user.click(await screen.findByRole("combobox", { name: "Repeat preset" }))
    await user.click(await screen.findByText("Custom schedule"))
    fireEvent.change(screen.getByRole("textbox", { name: "Custom cron" }), { target: { value: "" } })
    fireEvent.change(screen.getByRole("textbox", { name: "Timezone" }), { target: { value: "" } })
    fireEvent.click(await screen.findByRole("button", { name: "Save reminder" }))

    expect(await screen.findByText("Cron is required for recurring reminders")).toBeInTheDocument()
    expect(screen.getByText("Timezone is required for recurring reminders")).toBeInTheDocument()
    expect(mocks.createScheduledTaskReminder).not.toHaveBeenCalled()
  }, SLOW_SCHEDULE_FORM_TIMEOUT_MS)

  it("does not create a recurring reminder with scheduler-invalid cron or timezone", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=create&template=reminder")

    await user.type(await screen.findByRole("textbox", { name: "Title" }), "Invalid recurring reminder")
    fireEvent.click(await screen.findByText("Repeat"))
    await user.click(await screen.findByRole("combobox", { name: "Repeat preset" }))
    await user.click(await screen.findByText("Custom schedule"))
    fireEvent.change(screen.getByRole("textbox", { name: "Custom cron" }), {
      target: { value: "99 99 * * *" }
    })
    fireEvent.change(screen.getByRole("textbox", { name: "Timezone" }), {
      target: { value: "Mars/Olympus" }
    })
    await user.click(screen.getByRole("button", { name: "Save reminder" }))

    await waitFor(() => {
      expect(mocks.createScheduledTaskReminder).not.toHaveBeenCalled()
    })
    expect(screen.getAllByText("Cron minute must be between 0 and 59.").length).toBeGreaterThan(0)
    expect(screen.getByText("Timezone must be a valid IANA timezone.")).toBeInTheDocument()
  }, SLOW_SCHEDULE_FORM_TIMEOUT_MS)

  it("does not create a one-time reminder with whitespace-only run_at", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=create&template=reminder")

    await user.type(await screen.findByRole("textbox", { name: "Title" }), "Whitespace run at")
    fireEvent.change(screen.getByLabelText("Run once at"), { target: { value: "   " } })
    await user.click(await screen.findByRole("button", { name: "Save reminder" }))

    await waitFor(() => {
      expect(mocks.createScheduledTaskReminder).not.toHaveBeenCalled()
    })
    expect(screen.getByText("Run at is required for one-time reminders")).toBeInTheDocument()
  })

  it("does not create a recurring reminder with whitespace-only cron and timezone", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=create&template=reminder")

    fireEvent.change(await screen.findByRole("textbox", { name: "Title" }), {
      target: { value: "Whitespace recurring reminder" }
    })
    fireEvent.click(await screen.findByText("Repeat"))
    await user.click(await screen.findByRole("combobox", { name: "Repeat preset" }))
    await user.click(await screen.findByText("Custom schedule"))
    fireEvent.change(screen.getByRole("textbox", { name: "Custom cron" }), { target: { value: "   " } })
    fireEvent.change(screen.getByRole("textbox", { name: "Timezone" }), { target: { value: "   " } })
    await user.click(screen.getByRole("button", { name: "Save reminder" }))

    await waitFor(() => {
      expect(mocks.createScheduledTaskReminder).not.toHaveBeenCalled()
    })
    expect(screen.getByText("Cron is required for recurring reminders")).toBeInTheDocument()
    expect(screen.getByText("Timezone is required for recurring reminders")).toBeInTheDocument()
  }, SLOW_SCHEDULE_FORM_TIMEOUT_MS)

  it("preserves an existing recurring custom cron when editing unrelated fields", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [
        {
          id: "reminder_task:custom",
          primitive: "reminder_task",
          title: "Monday report",
          description: "Review generated digest",
          status: "scheduled",
          enabled: true,
          schedule_summary: "Every Monday at 09:00",
          timezone: "America/Los_Angeles",
          next_run_at: "2026-03-09T16:00:00+00:00",
          last_run_at: null,
          edit_mode: "native",
          manage_url: null,
          source_ref: {
            task_id: "custom",
            schedule_kind: "recurring",
            cron: "*/15 9 * * mon",
            timezone: "America/Los_Angeles"
          }
        }
      ],
      total: 1,
      partial: false,
      errors: []
    })
    mocks.updateScheduledTaskReminder.mockResolvedValue({
      id: "reminder_task:custom",
      primitive: "reminder_task",
      title: "Updated Monday report",
      description: "Review generated digest",
      status: "scheduled",
      enabled: true,
      edit_mode: "native",
      manage_url: null,
      source_ref: { task_id: "custom" }
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    expect(await screen.findByText("Monday report")).toBeInTheDocument()
    fireEvent.click(await screen.findByRole("button", { name: "Edit Monday report" }))
    expect(await screen.findByText("Edit reminder")).toBeInTheDocument()
    fireEvent.change(await screen.findByRole("textbox", { name: "Title" }), {
      target: { value: "Updated Monday report" }
    })
    fireEvent.click(await screen.findByRole("button", { name: "Save reminder" }))

    await waitFor(() => {
      expect(mocks.updateScheduledTaskReminder).toHaveBeenCalledWith(
        "reminder_task:custom",
        expect.objectContaining({
          title: "Updated Monday report",
          schedule_kind: "recurring",
          cron: "*/15 9 * * mon",
          timezone: "America/Los_Angeles"
        })
      )
    })
  })

  it("edits and deletes a reminder task from the table", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [
        {
          id: "reminder_task:1",
          primitive: "reminder_task",
          title: "Review notes",
          description: "Check the backlog",
          status: "scheduled",
          enabled: true,
          schedule_summary: "2026-03-21T09:00:00+00:00",
          timezone: "UTC",
          next_run_at: "2026-03-21T09:00:00+00:00",
          last_run_at: null,
          edit_mode: "native",
          manage_url: null,
          source_ref: { task_id: "1", schedule_kind: "one_time", run_at: "2026-03-21T09:00:00+00:00" }
        }
      ],
      total: 1,
      partial: false,
      errors: []
    })
    mocks.updateScheduledTaskReminder.mockResolvedValue({
      id: "reminder_task:1",
      primitive: "reminder_task",
      title: "Updated review",
      description: "Check the backlog",
      status: "scheduled",
      enabled: true,
      edit_mode: "native",
      manage_url: null,
      source_ref: { task_id: "1" }
    })
    mocks.deleteScheduledTaskReminder.mockResolvedValue({ deleted: true })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    expect(await screen.findByText("Review notes")).toBeInTheDocument()
    fireEvent.click(await screen.findByRole("button", { name: "Edit Review notes" }))
    expect(await screen.findByText("Edit reminder")).toBeInTheDocument()
    fireEvent.change(await screen.findByRole("textbox", { name: "Title" }), {
      target: { value: "Updated review" }
    })
    fireEvent.click(await screen.findByRole("button", { name: "Save reminder" }))

    await waitFor(() => {
      expect(mocks.updateScheduledTaskReminder).toHaveBeenCalledWith(
        "reminder_task:1",
        expect.objectContaining({ title: "Updated review" })
      )
    })

    fireEvent.click(await screen.findByRole("button", { name: "Delete Review notes" }))

    await waitFor(() => {
      expect(mocks.deleteScheduledTaskReminder).toHaveBeenCalledWith("reminder_task:1")
    })
  })
})
