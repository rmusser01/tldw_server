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

import { ScheduledTasksPage } from "../ScheduledTasksPage"

const fetchMock = vi.fn()
vi.stubGlobal("fetch", fetchMock)

const renderWithQueryClient = (ui: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })

  return render(
    <MemoryRouter initialEntries={["/scheduled-tasks"]}>
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

    renderWithQueryClient(<ScheduledTasksPage />)

    expect(
      await screen.findByRole("heading", {
        name: "Scheduled tasks are unavailable on this server"
      })
    ).toBeInTheDocument()
    expect(
      screen.getByText("The connected server does not advertise scheduled task management.")
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Diagnostics")).toHaveTextContent("/api/v1/scheduled-tasks")
    expect(mocks.listScheduledTasks).not.toHaveBeenCalled()
  })

  it("shows auth-required recovery copy for scheduled task load failures", async () => {
    mocks.listScheduledTasks.mockRejectedValue(
      Object.assign(new Error("Request failed: 401 (GET /api/v1/scheduled-tasks)"), {
        status: 401
      })
    )

    renderWithQueryClient(<ScheduledTasksPage />)

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
  })

  it("keeps partial scheduled task errors behind diagnostics", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: true,
      errors: ["Watchlist jobs failed at /api/v1/watchlists/jobs"]
    })

    renderWithQueryClient(<ScheduledTasksPage />)

    expect(
      await screen.findByRole("heading", {
        name: "Scheduled tasks are partially available"
      })
    ).toBeInTheDocument()
    expect(
      screen.getByText("Some scheduled task data loaded, but one dependency could not be reached.")
    ).toBeInTheDocument()

    const diagnostics = screen.getByLabelText("Diagnostics")
    expect(within(diagnostics).getByText("Watchlist jobs failed at /api/v1/watchlists/jobs")).toBeInTheDocument()
  })

  it("renders reminder rows as native CRUD and watchlist rows as external-managed", async () => {
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
          source_ref: { task_id: "1" }
        },
        {
          id: "watchlist_job:2",
          primitive: "watchlist_job",
          title: "Morning digest",
          description: "Watchlist run",
          status: "scheduled",
          enabled: true,
          schedule_summary: "0 9 * * *",
          timezone: "UTC",
          next_run_at: "2026-03-21T09:00:00+00:00",
          last_run_at: null,
          edit_mode: "external",
          manage_url: "/watchlists?tab=jobs",
          source_ref: { job_id: 2 }
        }
      ],
      total: 2,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />)

    expect(await screen.findByRole("heading", { level: 2, name: "Scheduled tasks" })).toBeInTheDocument()
    expect(await screen.findByText("Review notes")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create Reminder Task" })).toBeInTheDocument()
    expect(await screen.findByRole("button", { name: "Edit" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Delete" })).toBeInTheDocument()
    expect(await screen.findByText("Morning digest")).toBeInTheDocument()
    expect(await screen.findByRole("link", { name: "Manage in Watchlists" })).toHaveAttribute(
      "href",
      "/watchlists?tab=jobs"
    )
    expect(screen.queryByRole("button", { name: "Edit watchlist job" })).not.toBeInTheDocument()
  })

  it("creates a reminder task from the editor and refreshes the list", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
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

    renderWithQueryClient(<ScheduledTasksPage />)

    fireEvent.click(await screen.findByRole("button", { name: "Create Reminder Task" }))
    await user.type(await screen.findByRole("textbox", { name: "Title" }), "Daily review")
    await user.type(screen.getByRole("textbox", { name: "Run at" }), "2026-03-21T10:00:00+00:00")
    await user.click(await screen.findByRole("button", { name: "Save Reminder Task" }))

    await waitFor(() => {
      expect(mocks.createScheduledTaskReminder).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Daily review",
          schedule_kind: "one_time",
          run_at: "2026-03-21T10:00:00+00:00",
          enabled: true
        })
      )
    })
  })

  it("does not create a one-time reminder without run_at", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />)

    await user.click(await screen.findByRole("button", { name: "Create Reminder Task" }))
    await user.type(await screen.findByRole("textbox", { name: "Title" }), "Missing run at")
    await user.click(await screen.findByRole("button", { name: "Save Reminder Task" }))

    await waitFor(() => {
      expect(mocks.createScheduledTaskReminder).not.toHaveBeenCalled()
    })
    expect(screen.getByText("Run at is required for one-time reminders")).toBeInTheDocument()
  })

  it("does not create a recurring reminder without cron and timezone", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />)

    await user.click(await screen.findByRole("button", { name: "Create Reminder Task" }))
    await user.type(await screen.findByRole("textbox", { name: "Title" }), "Recurring reminder")
    await user.click(await screen.findByRole("combobox", { name: "Schedule kind" }))
    await user.click(await screen.findByText("Recurring"))
    fireEvent.click(await screen.findByRole("button", { name: "Save Reminder Task" }))

    expect(await screen.findByText("Cron is required for recurring reminders")).toBeInTheDocument()
    expect(screen.getByText("Timezone is required for recurring reminders")).toBeInTheDocument()
    expect(mocks.createScheduledTaskReminder).not.toHaveBeenCalled()
  })

  it("does not create a one-time reminder with whitespace-only run_at", async () => {
    const user = userEvent.setup()

    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />)

    await user.click(await screen.findByRole("button", { name: "Create Reminder Task" }))
    await user.type(await screen.findByRole("textbox", { name: "Title" }), "Whitespace run at")
    fireEvent.change(screen.getByRole("textbox", { name: "Run at" }), { target: { value: "   " } })
    await user.click(await screen.findByRole("button", { name: "Save Reminder Task" }))

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

    renderWithQueryClient(<ScheduledTasksPage />)

    fireEvent.click(await screen.findByRole("button", { name: "Create Reminder Task" }))
    fireEvent.change(screen.getByRole("textbox", { name: "Title" }), {
      target: { value: "Whitespace recurring reminder" }
    })
    await user.click(await screen.findByRole("combobox", { name: "Schedule kind" }))
    await user.click(await screen.findByText("Recurring"))
    fireEvent.change(screen.getByRole("textbox", { name: "Cron" }), { target: { value: "   " } })
    fireEvent.change(screen.getByRole("textbox", { name: "Timezone" }), { target: { value: "   " } })
    await user.click(screen.getByRole("button", { name: "Save Reminder Task" }))

    await waitFor(() => {
      expect(mocks.createScheduledTaskReminder).not.toHaveBeenCalled()
    })
    expect(screen.getByText("Cron is required for recurring reminders")).toBeInTheDocument()
    expect(screen.getByText("Timezone is required for recurring reminders")).toBeInTheDocument()
  }, 10000)

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

    renderWithQueryClient(<ScheduledTasksPage />)

    expect(await screen.findByText("Review notes")).toBeInTheDocument()
    fireEvent.click(await screen.findByRole("button", { name: "Edit" }))
    expect(await screen.findByText("Edit reminder task")).toBeInTheDocument()
    fireEvent.change(await screen.findByRole("textbox", { name: "Title" }), {
      target: { value: "Updated review" }
    })
    fireEvent.click(await screen.findByRole("button", { name: "Save Reminder Task" }))

    await waitFor(() => {
      expect(mocks.updateScheduledTaskReminder).toHaveBeenCalledWith(
        "reminder_task:1",
        expect.objectContaining({ title: "Updated review" })
      )
    })

    fireEvent.click(await screen.findByRole("button", { name: "Delete" }))

    await waitFor(() => {
      expect(mocks.deleteScheduledTaskReminder).toHaveBeenCalledWith("reminder_task:1")
    })
  })
})
