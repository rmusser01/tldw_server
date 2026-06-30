// @vitest-environment jsdom

import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { expectInsideDesignSystemAlert } from "@/test-utils/designSystemAlert"

const mocks = vi.hoisted(() => ({
  useCanonicalConnectionConfig: vi.fn(),
  listScheduledTasks: vi.fn(),
  getScheduledTaskCapabilities: vi.fn(),
  createScheduledTaskPreview: vi.fn(),
  createScheduledTaskDefinition: vi.fn(),
  updateScheduledTaskDefinition: vi.fn(),
  getScheduledTaskDefinition: vi.fn(),
  listScheduledTaskPreviews: vi.fn(),
  listScheduledTaskDefinitionAudit: vi.fn(),
  pauseScheduledTaskDefinition: vi.fn(),
  resumeScheduledTaskDefinition: vi.fn(),
  archiveScheduledTaskDefinition: vi.fn(),
  duplicateScheduledTaskDefinition: vi.fn(),
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
  getScheduledTaskCapabilities: (...args: unknown[]) =>
    mocks.getScheduledTaskCapabilities(...args),
  createScheduledTaskPreview: (...args: unknown[]) =>
    mocks.createScheduledTaskPreview(...args),
  createScheduledTaskDefinition: (...args: unknown[]) =>
    mocks.createScheduledTaskDefinition(...args),
  updateScheduledTaskDefinition: (...args: unknown[]) =>
    mocks.updateScheduledTaskDefinition(...args),
  getScheduledTaskDefinition: (...args: unknown[]) =>
    mocks.getScheduledTaskDefinition(...args),
  listScheduledTaskPreviews: (...args: unknown[]) =>
    mocks.listScheduledTaskPreviews(...args),
  listScheduledTaskDefinitionAudit: (...args: unknown[]) =>
    mocks.listScheduledTaskDefinitionAudit(...args),
  pauseScheduledTaskDefinition: (...args: unknown[]) =>
    mocks.pauseScheduledTaskDefinition(...args),
  resumeScheduledTaskDefinition: (...args: unknown[]) =>
    mocks.resumeScheduledTaskDefinition(...args),
  archiveScheduledTaskDefinition: (...args: unknown[]) =>
    mocks.archiveScheduledTaskDefinition(...args),
  duplicateScheduledTaskDefinition: (...args: unknown[]) =>
    mocks.duplicateScheduledTaskDefinition(...args),
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

const availableAutomationCapabilities = {
  items: [
    {
      family: "recurring_question",
      family_availability: "available",
      actions: {
        create_definition: {
          status: "available",
          reason: null,
          required_permissions: []
        },
        update_definition: {
          status: "available",
          reason: null,
          required_permissions: []
        }
      },
      missing_dependencies: [],
      related_capabilities: {},
      schema_version: "2026-06-10"
    },
    {
      family: "agent_task",
      family_availability: "available",
      actions: {
        create_definition: {
          status: "available",
          reason: null,
          required_permissions: []
        }
      },
      missing_dependencies: [],
      related_capabilities: {},
      schema_version: "2026-06-10"
    }
  ]
}

const emptyAutomationCapabilities = {
  items: []
}

const automationTask = (overrides: Record<string, unknown> = {}) => ({
  id: "automation_definition:definition_1",
  primitive: "automation_definition",
  title: "Track answer",
  description: "Ask until the answer appears",
  status: "configured_execution_unavailable",
  enabled: true,
  schedule_summary: "0 9 * * *",
  timezone: "UTC",
  next_run_at: null,
  last_run_at: null,
  edit_mode: "native",
  manage_url: null,
  source_ref: {
    definition_id: "definition_1",
    family: "recurring_question",
    lifecycle: "configured",
    health: "execution_unavailable",
    visibility: "private"
  },
  ...overrides
})

const definitionResponse = (overrides: Record<string, unknown> = {}) => ({
  id: "definition_1",
  version: 1,
  family: "recurring_question",
  name: "Track answer",
  description: null,
  lifecycle: "configured",
  health: "execution_unavailable",
  schedule: {},
  input: {},
  config: {},
  visibility_policy: { visibility: "private" },
  notification_policy: {},
  approval_policy: {},
  ...overrides
})

const expectInsideDesignSystemComponent = (
  text: string | RegExp,
  componentName: string
): HTMLElement => {
  const marker = `[data-ds-component="${componentName}"]`
  const match = screen
    .getAllByText(text)
    .map((node) => node.closest(marker))
    .find((node): node is HTMLElement => node instanceof HTMLElement)

  expect(match).toBeTruthy()
  return match
}

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
    mocks.getScheduledTaskCapabilities.mockResolvedValue(emptyAutomationCapabilities)
    mocks.createScheduledTaskPreview.mockResolvedValue({
      id: "preview_1",
      mode: "create",
      family: "recurring_question",
      status: "valid",
      normalized_config: {},
      validation_errors: [],
      warnings: [],
      visibility_policy: {},
      schedule_preview: {},
      redaction_policy: {},
      expires_at: "2026-06-10T00:00:00Z"
    })
    mocks.createScheduledTaskDefinition.mockResolvedValue(definitionResponse())
    mocks.updateScheduledTaskDefinition.mockResolvedValue(definitionResponse())
    mocks.getScheduledTaskDefinition.mockResolvedValue(definitionResponse())
    mocks.listScheduledTaskPreviews.mockResolvedValue({
      items: [],
      total: 0,
      limit: 20,
      offset: 0,
      has_more: false
    })
    mocks.listScheduledTaskDefinitionAudit.mockResolvedValue({
      items: [],
      total: 0,
      limit: 20,
      offset: 0,
      has_more: false
    })
    mocks.pauseScheduledTaskDefinition.mockResolvedValue(definitionResponse({ lifecycle: "paused" }))
    mocks.resumeScheduledTaskDefinition.mockResolvedValue(definitionResponse({ lifecycle: "configured" }))
    mocks.archiveScheduledTaskDefinition.mockResolvedValue(definitionResponse({ lifecycle: "archived" }))
    mocks.duplicateScheduledTaskDefinition.mockResolvedValue(
      definitionResponse({ id: "definition_copy", lifecycle: "paused" })
    )
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
    expect(screen.getByText("Future scheduled questions and agent outputs appear here only when the results API and each task visibility policy route them here.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Open signal for Release monitor" })).toBeInTheDocument()
    expect(screen.getByText("Found 3 results from Release feed.")).toBeInTheDocument()
  })

  it("opens the Results tab from the alias path and opens the result drawer", async () => {
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
    expect(await screen.findByRole("dialog", { name: /Release monitor/i })).toBeInTheDocument()
    expect(screen.queryByText("Result signal not found.")).not.toBeInTheDocument()
  })

  it("can navigate from the Results alias path back to Overview", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })
    const user = userEvent.setup()

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks/results")

    expect(await screen.findByRole("tab", { name: "Results" })).toHaveAttribute(
      "aria-selected",
      "true"
    )

    await user.click(screen.getByRole("tab", { name: "Overview" }))

    await waitFor(() => {
      expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute(
        "aria-selected",
        "true"
      )
    })
    expect(screen.getByText("Total scheduled tasks")).toBeInTheDocument()
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
    expectInsideDesignSystemAlert("Result signal not found.")
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

  it("opens the Create tab with the planned Recurring Question shell from the route", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(
      <ScheduledTasksPage />,
      "/scheduled-tasks?tab=create&template=recurring_question"
    )

    expect(await screen.findByRole("tab", { name: "Create" })).toHaveAttribute("aria-selected", "true")
    expect(
      await screen.findByText(
        "Recurring Question scheduling is planned for the API contract and is not executable in this client yet."
      )
    ).toBeInTheDocument()
    expect(screen.getByText("Scheduled RAG query support")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Open Research" })).toHaveAttribute(
      "href",
      "/research"
    )
    expect(screen.queryByRole("button", { name: /Create/i })).not.toBeInTheDocument()
  })

  it("opens the Create tab with the planned Agent Task shell from the route", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })

    renderWithQueryClient(
      <ScheduledTasksPage />,
      "/scheduled-tasks?tab=create&template=agent_task"
    )

    expect(await screen.findByRole("tab", { name: "Create" })).toHaveAttribute("aria-selected", "true")
    expect(
      await screen.findByText(
        "Agent Task scheduling is planned for the API contract and is not executable in this client yet."
      )
    ).toBeInTheDocument()
    expect(screen.getByText("Preview and risk classification")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Open ACP Playground" })).toHaveAttribute(
      "href",
      "/acp-playground"
    )
    expect(screen.queryByRole("button", { name: /Create/i })).not.toBeInTheDocument()
  })

  it("loads automation capabilities for API-first create flows", async () => {
    mocks.listScheduledTasks.mockResolvedValue({
      items: [],
      total: 0,
      partial: false,
      errors: []
    })
    mocks.getScheduledTaskCapabilities.mockResolvedValue(availableAutomationCapabilities)

    renderWithQueryClient(
      <ScheduledTasksPage />,
      "/scheduled-tasks?tab=create&template=recurring_question"
    )

    expect(await screen.findByText("Create Recurring question")).toBeInTheDocument()
    expect(mocks.getScheduledTaskCapabilities).toHaveBeenCalledTimes(1)
    expect(screen.getByLabelText("Question")).toBeInTheDocument()
  })

  it("creates a Recurring Question definition through preview and create APIs", async () => {
    const user = userEvent.setup()
    mocks.getScheduledTaskCapabilities.mockResolvedValue(availableAutomationCapabilities)
    mocks.listScheduledTasks
      .mockResolvedValueOnce({
        items: [],
        total: 0,
        partial: false,
        errors: []
      })
      .mockResolvedValueOnce({
        items: [automationTask()],
        total: 1,
        partial: false,
        errors: []
      })
    mocks.createScheduledTaskPreview.mockResolvedValue({
      id: "preview_recurring",
      mode: "create",
      family: "recurring_question",
      status: "valid",
      normalized_config: { name: "Track answer" },
      validation_errors: [],
      warnings: [],
      visibility_policy: { visibility: "private" },
      schedule_preview: { summary: "Manual" },
      redaction_policy: {},
      expires_at: "2026-06-10T00:00:00Z"
    })
    mocks.createScheduledTaskDefinition.mockResolvedValue(definitionResponse())

    renderWithQueryClient(
      <ScheduledTasksPage />,
      "/scheduled-tasks?tab=create&template=recurring_question"
    )

    await user.type(await screen.findByLabelText("Question"), "Has the answer appeared?")
    await user.click(screen.getByRole("button", { name: "Preview" }))
    expect(await screen.findByText("Preview ready")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Save definition" }))

    await waitFor(() => {
      expect(mocks.createScheduledTaskPreview).toHaveBeenCalledWith(
        expect.objectContaining({
          family: "recurring_question",
          input: expect.objectContaining({
            question: "Has the answer appeared?"
          })
        })
      )
    })
    expect(mocks.createScheduledTaskDefinition).toHaveBeenCalledWith({
      preview_id: "preview_recurring",
      initial_lifecycle: "configured"
    })
    await waitFor(() => expect(mocks.listScheduledTasks).toHaveBeenCalledTimes(2))
    await waitFor(() => {
      expect(screen.getAllByText("Track answer").length).toBeGreaterThan(0)
    })
  })

  it("updates an automation definition through preview and update APIs", async () => {
    const user = userEvent.setup()
    mocks.getScheduledTaskCapabilities.mockResolvedValue(availableAutomationCapabilities)
    mocks.listScheduledTasks
      .mockResolvedValueOnce({
        items: [automationTask()],
        total: 1,
        partial: false,
        errors: []
      })
      .mockResolvedValueOnce({
        items: [
          automationTask({
            title: "Track answer updated",
            description: "Updated definition"
          })
        ],
        total: 1,
        partial: false,
        errors: []
      })
    mocks.getScheduledTaskDefinition.mockResolvedValue(
      definitionResponse({
        version: 3,
        input: {
          question: "Has the answer appeared?",
          success_criteria: "Answer found",
          scope: { collection_id: "research" }
        },
        schedule: { kind: "cron", cron: "0 9 * * *", timezone: "UTC" },
        visibility_policy: { visibility: "private" }
      })
    )
    mocks.createScheduledTaskPreview.mockResolvedValue({
      id: "preview_update",
      mode: "update",
      family: "recurring_question",
      definition_id: "definition_1",
      definition_version: 3,
      status: "valid",
      normalized_config: { name: "Track answer updated" },
      validation_errors: [],
      warnings: [],
      visibility_policy: { visibility: "private" },
      schedule_preview: { summary: "0 9 * * *" },
      redaction_policy: {},
      expires_at: "2026-06-10T00:00:00Z"
    })
    mocks.updateScheduledTaskDefinition.mockResolvedValue(
      definitionResponse({
        name: "Track answer updated",
        version: 4
      })
    )

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    await user.click(await screen.findByRole("button", { name: "Edit Track answer" }))
    expect(await screen.findByText("Update Recurring question")).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText("Question"), {
      target: { value: "Has the answer appeared now?" }
    })
    await user.click(screen.getByRole("button", { name: "Preview" }))
    expect(await screen.findByText("Preview ready")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Save definition" }))

    await waitFor(() => {
      expect(mocks.createScheduledTaskPreview).toHaveBeenCalledWith(
        expect.objectContaining({
          mode: "update",
          family: "recurring_question",
          definition_id: "definition_1",
          definition_version: 3,
          input: expect.objectContaining({
            question: "Has the answer appeared now?"
          })
        })
      )
    })
    expect(mocks.updateScheduledTaskDefinition).toHaveBeenCalledWith("definition_1", {
      preview_id: "preview_update"
    })
    await waitFor(() => expect(mocks.listScheduledTasks).toHaveBeenCalledTimes(2))
  }, SLOW_SCHEDULE_FORM_TIMEOUT_MS)

  it.each(["daily", "weekly", "interval", "one_time"])(
    "preserves %s schedule kind when updating an automation definition",
    async (scheduleKind) => {
      const user = userEvent.setup()
      mocks.getScheduledTaskCapabilities.mockResolvedValue(availableAutomationCapabilities)
      mocks.listScheduledTasks.mockResolvedValue({
        items: [automationTask()],
        total: 1,
        partial: false,
        errors: []
      })
      mocks.getScheduledTaskDefinition.mockResolvedValue(
        definitionResponse({
          version: 5,
          input: {
            question: "Has the answer appeared?",
            scope: { collection_id: "research" }
          },
          schedule: {
            kind: scheduleKind,
            timezone: "America/New_York",
            run_at: "2030-04-05T12:30:00Z",
            every_seconds: 3600,
            day_of_week: "monday"
          }
        })
      )
      mocks.createScheduledTaskPreview.mockResolvedValue({
        id: `preview_${scheduleKind}`,
        mode: "update",
        family: "recurring_question",
        definition_id: "definition_1",
        definition_version: 5,
        status: "valid",
        normalized_config: { name: "Track answer" },
        validation_errors: [],
        warnings: [],
        visibility_policy: { visibility: "private" },
        schedule_preview: { summary: scheduleKind },
        redaction_policy: {},
        expires_at: "2026-06-10T00:00:00Z"
      })

      renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

      await user.click(await screen.findByRole("button", { name: "Edit Track answer" }))
      expect(await screen.findByText("Update Recurring question")).toBeInTheDocument()
      await user.click(screen.getByRole("button", { name: "Preview" }))

      await waitFor(() => {
        expect(mocks.createScheduledTaskPreview).toHaveBeenCalledWith(
          expect.objectContaining({
            mode: "update",
            definition_id: "definition_1",
            schedule: expect.objectContaining({
              kind: scheduleKind,
              timezone: "America/New_York"
            })
          })
        )
      })
    },
    SLOW_SCHEDULE_FORM_TIMEOUT_MS
  )

  it("preserves an Agent Task string agent ref when updating", async () => {
    const user = userEvent.setup()
    mocks.getScheduledTaskCapabilities.mockResolvedValue(availableAutomationCapabilities)
    mocks.listScheduledTasks.mockResolvedValue({
      items: [
        automationTask({
          title: "Dispatch agent",
          source_ref: {
            definition_id: "agent_definition",
            family: "agent_task",
            lifecycle: "configured",
            health: "execution_unavailable",
            visibility: "private"
          }
        })
      ],
      total: 1,
      partial: false,
      errors: []
    })
    mocks.getScheduledTaskDefinition.mockResolvedValue(
      definitionResponse({
        id: "agent_definition",
        version: 2,
        family: "agent_task",
        name: "Dispatch agent",
        input: {
          agent_ref: "agent://primary",
          message: "Summarize the report"
        },
        schedule: { kind: "daily", timezone: "UTC" }
      })
    )
    mocks.createScheduledTaskPreview.mockResolvedValue({
      id: "preview_agent_update",
      mode: "update",
      family: "agent_task",
      definition_id: "agent_definition",
      definition_version: 2,
      status: "valid",
      normalized_config: { name: "Dispatch agent" },
      validation_errors: [],
      warnings: [],
      visibility_policy: { visibility: "private" },
      schedule_preview: { summary: "Daily" },
      redaction_policy: {},
      expires_at: "2026-06-10T00:00:00Z"
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    await user.click(await screen.findByRole("button", { name: "Edit Dispatch agent" }))
    expect(await screen.findByDisplayValue("agent://primary")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Preview" }))

    await waitFor(() => {
      expect(mocks.createScheduledTaskPreview).toHaveBeenCalledWith(
        expect.objectContaining({
          family: "agent_task",
          definition_id: "agent_definition",
          input: expect.objectContaining({
            agent_ref: "agent://primary",
            message: "Summarize the report"
          })
        })
      )
    })
  }, SLOW_SCHEDULE_FORM_TIMEOUT_MS)

  it("runs automation definition lifecycle actions from rows and refreshes the list", async () => {
    const user = userEvent.setup()
    mocks.getScheduledTaskCapabilities.mockResolvedValue(availableAutomationCapabilities)
    mocks.listScheduledTasks.mockResolvedValue({
      items: [
        automationTask({ id: "automation_definition:projected_definition_1" }),
        automationTask({
          id: "automation_definition:paused",
          title: "Paused agent",
          status: "paused",
          source_ref: {
            definition_id: "paused",
            family: "agent_task",
            lifecycle: "paused",
            health: "execution_unavailable"
          }
        })
      ],
      total: 2,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    await user.click(await screen.findByRole("button", { name: "Pause Track answer" }))
    await waitFor(() => {
      expect(mocks.pauseScheduledTaskDefinition).toHaveBeenCalledWith("definition_1")
    })

    await user.click(await screen.findByRole("button", { name: "Resume Paused agent" }))
    await waitFor(() => {
      expect(mocks.resumeScheduledTaskDefinition).toHaveBeenCalledWith("paused")
    })

    await user.click(await screen.findByRole("button", { name: "Archive Track answer" }))
    await waitFor(() => {
      expect(mocks.archiveScheduledTaskDefinition).toHaveBeenCalledWith("definition_1")
    })

    await user.click(await screen.findByRole("button", { name: "Duplicate Track answer" }))
    await waitFor(() => {
      expect(mocks.duplicateScheduledTaskDefinition).toHaveBeenCalledWith(
        "definition_1",
        expect.objectContaining({ name: expect.stringContaining("Track answer") })
      )
    })
    await waitFor(() => expect(mocks.listScheduledTasks).toHaveBeenCalledTimes(5))
  }, SLOW_SCHEDULE_FORM_TIMEOUT_MS)

  it("shows API error detail code and message for automation lifecycle failures", async () => {
    const user = userEvent.setup()
    mocks.getScheduledTaskCapabilities.mockResolvedValue(availableAutomationCapabilities)
    mocks.listScheduledTasks.mockResolvedValue({
      items: [automationTask()],
      total: 1,
      partial: false,
      errors: []
    })
    mocks.pauseScheduledTaskDefinition.mockRejectedValue({
      details: {
        detail: {
          code: "definition_locked",
          message: "Definition is locked by policy."
        }
      }
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    await user.click(await screen.findByRole("button", { name: "Pause Track answer" }))

    await waitFor(() => {
      expect(
        screen.getAllByText("definition_locked: Definition is locked by policy.").length
      ).toBeGreaterThan(0)
    })
  })

  it("shows row Results buttons only for real result signals", async () => {
    mocks.getScheduledTaskCapabilities.mockResolvedValue(availableAutomationCapabilities)
    mocks.listScheduledTasks.mockResolvedValue({
      items: [
        {
          id: "reminder_task:completed",
          primitive: "reminder_task",
          title: "Completed review",
          description: "No output was produced",
          status: "completed",
          enabled: true,
          schedule_summary: "One-time reminder",
          timezone: "UTC",
          next_run_at: null,
          last_run_at: "2030-04-05T12:30:00+00:00",
          edit_mode: "native",
          manage_url: null,
          source_ref: { task_id: "completed" }
        },
        automationTask({
          id: "automation_definition:no_result",
          title: "Automation no result",
          source_ref: {
            definition_id: "no_result",
            family: "recurring_question",
            lifecycle: "configured",
            health: "execution_unavailable"
          }
        }),
        automationTask({
          id: "automation_definition:with_result",
          title: "Automation result",
          status: "configured",
          source_ref: {
            definition_id: "with_result",
            family: "recurring_question",
            lifecycle: "configured",
            health: "ready",
            latest_result_id: "909",
            result_count: 1
          }
        })
      ],
      total: 3,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    expect(await screen.findByText("Completed review")).toBeInTheDocument()
    expect(screen.getByText("Automation no result")).toBeInTheDocument()
    expect(screen.getByText("Automation result")).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "View results for Completed review" })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "View results for Automation no result" })
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "View results for Automation result" })
    ).toBeInTheDocument()
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
    expectInsideDesignSystemAlert("That tab is not available. Showing Overview.")
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
    expectInsideDesignSystemAlert("Task not found.")
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
    expect(screen.getByText("Latest result signal")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Open latest result signal" })).toBeInTheDocument()
    expect(screen.getAllByText(/2030/).length).toBeGreaterThan(0)
    expect(screen.getByText(/Watchlists remains the full workspace/)).toBeInTheDocument()
    expectInsideDesignSystemAlert(/Watchlists remains the full workspace/)
    expectInsideDesignSystemComponent("Review required", "Badge")
    expectInsideDesignSystemComponent("Active", "Badge")
    expect(screen.queryByRole("columnheader", { name: "Task" })).not.toBeInTheDocument()
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
    expect(
      within(reminderRow as HTMLElement)
        .getByText("Needs attention")
        .closest('[data-ds-component="Badge"]')
    ).not.toBeNull()
    expect(within(reminderRow as HTMLElement).getByText("No completed runs yet")).toBeInTheDocument()

    expect(screen.getByRole("button", { name: "Inspect Review notes" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "View results for Review notes" })).toBeInTheDocument()
    expect(await screen.findByRole("button", { name: "Edit Review notes" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Delete Review notes" })).toBeInTheDocument()
    expect(await screen.findByText("Morning digest")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "View results for Morning digest" })).toBeInTheDocument()
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
  }, SLOW_SCHEDULE_FORM_TIMEOUT_MS)

  it("opens the automation editor from the detail drawer without leaving the drawer open", async () => {
    const user = userEvent.setup()

    mocks.getScheduledTaskCapabilities.mockResolvedValue(availableAutomationCapabilities)
    mocks.listScheduledTasks.mockResolvedValue({
      items: [automationTask()],
      total: 1,
      partial: false,
      errors: []
    })
    mocks.getScheduledTaskDefinition.mockResolvedValue(definitionResponse())

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    await user.click(await screen.findByRole("button", { name: "Inspect Track answer" }))
    expect(await screen.findByRole("dialog", { name: /Track answer/i })).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Edit definition" }))

    expect(await screen.findByText("Update Recurring question")).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /Track answer/i })).not.toBeInTheDocument()
    })
  }, SLOW_SCHEDULE_FORM_TIMEOUT_MS)

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
  }, SLOW_SCHEDULE_FORM_TIMEOUT_MS)

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
        },
        automationTask({
          id: "automation_definition:archived",
          title: "Archived automation",
          status: "archived",
          source_ref: {
            definition_id: "archived",
            family: "recurring_question",
            lifecycle: "archived",
            health: "execution_unavailable"
          }
        })
      ],
      total: 3,
      partial: false,
      errors: []
    })

    renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

    expect(await screen.findByText("Healthy reminder")).toBeInTheDocument()
    expect(screen.getByText("Blocked monitor")).toBeInTheDocument()
    expect(screen.getByText("Archived automation")).toBeInTheDocument()
    const healthyRow = screen.getByText("Healthy reminder").closest("tr")
    expect(healthyRow).not.toBeNull()
    expect(within(healthyRow as HTMLElement).getByText("Waiting for next run")).toBeInTheDocument()
    expect(within(healthyRow as HTMLElement).queryByText("scheduled")).not.toBeInTheDocument()

    await user.click(screen.getByRole("combobox", { name: "Status filter" }))
    await user.click(await screen.findByTitle("Needs attention"))

    expect(screen.queryByText("Healthy reminder")).not.toBeInTheDocument()
    expect(screen.getByText("Blocked monitor")).toBeInTheDocument()
    expect(screen.queryByText("Archived automation")).not.toBeInTheDocument()

    await user.click(screen.getByRole("combobox", { name: "Status filter" }))
    await user.click(await screen.findByTitle("Archived"))

    expect(screen.queryByText("Healthy reminder")).not.toBeInTheDocument()
    expect(screen.queryByText("Blocked monitor")).not.toBeInTheDocument()
    expect(screen.getByText("Archived automation")).toBeInTheDocument()

    await user.click(screen.getByRole("combobox", { name: "Status filter" }))
    await user.click(await screen.findByTitle("All statuses"))
    fireEvent.change(screen.getByRole("textbox", { name: "Search scheduled tasks" }), {
      target: { value: "healthy" }
    })

    expect(screen.getByText("Healthy reminder")).toBeInTheDocument()
    expect(screen.queryByText("Blocked monitor")).not.toBeInTheDocument()
  }, SLOW_SCHEDULE_FORM_TIMEOUT_MS)

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
    expectInsideDesignSystemComponent("Loading tasks and latest run state", "LoadingState")
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
    expectInsideDesignSystemComponent("No scheduled tasks yet.", "EmptyState")
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
  }, SLOW_SCHEDULE_FORM_TIMEOUT_MS)

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
  }, SLOW_SCHEDULE_FORM_TIMEOUT_MS)

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
  }, SLOW_SCHEDULE_FORM_TIMEOUT_MS)

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
  }, SLOW_SCHEDULE_FORM_TIMEOUT_MS)

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
  }, SLOW_SCHEDULE_FORM_TIMEOUT_MS)

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
  }, SLOW_SCHEDULE_FORM_TIMEOUT_MS)
})
