import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args)
}))

import {
  archiveScheduledTaskDefinition,
  createScheduledTaskPreview,
  createScheduledTaskReminder,
  deleteScheduledTaskReminder,
  getScheduledTaskCapabilities,
  getScheduledTaskDefinition,
  getScheduledTask,
  listScheduledTaskDefinitionAudit,
  listScheduledTaskDefinitions,
  listScheduledTasks,
  pauseScheduledTaskDefinition,
  updateScheduledTaskReminder,
  type CreateScheduledTaskReminderPayload,
  type ScheduledTaskPreviewCreateRequest,
  type UpdateScheduledTaskReminderPayload
} from "@/services/scheduled-tasks-control-plane"

describe("scheduled-tasks control-plane contract", () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
  })

  it("lists normalized scheduled tasks including partial metadata", async () => {
    mocks.bgRequest.mockResolvedValue({
      items: [
        {
          id: "reminder_task:abc",
          primitive: "reminder_task",
          title: "Review notes",
          description: "Check the backlog",
          status: "scheduled",
          enabled: true,
          schedule_summary: "2026-03-21T09:00:00+00:00",
          timezone: null,
          next_run_at: "2026-03-21T09:00:00+00:00",
          last_run_at: null,
          edit_mode: "native",
          manage_url: null,
          source_ref: { task_id: "abc" }
        }
      ],
      total: 1,
      partial: false,
      errors: []
    })

    const response = await listScheduledTasks()

    expect(response.partial).toBe(false)
    expect(response.errors).toEqual([])
    expect(response.items[0]?.source_ref).toMatchObject({ task_id: "abc" })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        method: "GET",
        path: "/api/v1/scheduled-tasks"
      })
    )
  })

  it("fetches a single scheduled task by encoded id", async () => {
    mocks.bgRequest.mockResolvedValue({
      id: "reminder_task:abc",
      primitive: "reminder_task",
      title: "Review notes",
      status: "scheduled",
      enabled: true,
      edit_mode: "native",
      source_ref: { task_id: "abc" }
    })

    const response = await getScheduledTask("reminder_task:abc")

    expect(response.id).toBe("reminder_task:abc")
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        method: "GET",
        path: "/api/v1/scheduled-tasks/reminder_task%3Aabc"
      })
    )
  })

  it("creates reminder tasks with the full typed payload", async () => {
    const payload: CreateScheduledTaskReminderPayload = {
      title: "Follow up",
      body: "Send the update",
      schedule_kind: "one_time",
      run_at: "2026-03-21T10:00:00+00:00",
      enabled: true
    }
    mocks.bgRequest.mockResolvedValue({
      id: "reminder_task:abc",
      primitive: "reminder_task",
      title: "Follow up",
      status: "scheduled",
      enabled: true,
      edit_mode: "native",
      source_ref: { task_id: "abc", schedule_kind: "one_time" }
    })

    const response = await createScheduledTaskReminder(payload)

    expect(response.source_ref).toMatchObject({ task_id: "abc", schedule_kind: "one_time" })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        method: "POST",
        path: "/api/v1/scheduled-tasks/reminders",
        body: expect.objectContaining({
          title: "Follow up",
          body: "Send the update",
          schedule_kind: "one_time",
          run_at: "2026-03-21T10:00:00+00:00",
          enabled: true
        })
      })
    )
  })

  it("updates and deletes reminder tasks through reminder routes", async () => {
    const updatePayload: UpdateScheduledTaskReminderPayload = {
      enabled: false,
      title: "Updated follow up"
    }
    mocks.bgRequest
      .mockResolvedValueOnce({
        id: "reminder_task:abc",
        primitive: "reminder_task",
        title: "Updated follow up",
        status: "disabled",
        enabled: false,
        edit_mode: "native",
        source_ref: { task_id: "abc" }
      })
      .mockResolvedValueOnce({ deleted: true })

    const updated = await updateScheduledTaskReminder("reminder_task:abc", updatePayload)
    const deleted = await deleteScheduledTaskReminder("reminder_task:abc")

    expect(updated.enabled).toBe(false)
    expect(deleted.deleted).toBe(true)
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        method: "PATCH",
        path: "/api/v1/scheduled-tasks/reminders/abc",
        body: expect.objectContaining({
          enabled: false,
          title: "Updated follow up"
        })
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        method: "DELETE",
        path: "/api/v1/scheduled-tasks/reminders/abc"
      })
    )
  })

  it("rejects unsafe null reminder patch fields before sending the request", async () => {
    await expect(
      updateScheduledTaskReminder(
        "reminder_task:abc",
        {
          title: null,
          body: "Still unsafe"
        } as unknown as UpdateScheduledTaskReminderPayload
      )
    ).rejects.toThrow("title cannot be null")

    expect(mocks.bgRequest).not.toHaveBeenCalled()
  })

  it("fetches automation capabilities from the capabilities endpoint", async () => {
    mocks.bgRequest.mockResolvedValue({ items: [] })

    await getScheduledTaskCapabilities()

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        method: "GET",
        path: "/api/v1/scheduled-tasks/capabilities"
      })
    )
  })

  it("creates automation previews with body and optional idempotency header", async () => {
    const payload: ScheduledTaskPreviewCreateRequest = {
      family: "recurring_question",
      mode: "create",
      name: "Track answer",
      schedule: { kind: "cron", cron: "0 9 * * *" }
    }
    mocks.bgRequest.mockResolvedValue({ id: "preview_1" })

    await createScheduledTaskPreview(payload, { idempotencyKey: "preview-key-1" })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        method: "POST",
        path: "/api/v1/scheduled-tasks/previews",
        body: payload,
        headers: { "Idempotency-Key": "preview-key-1" }
      })
    )
  })

  it("encodes automation definition list query parameters", async () => {
    mocks.bgRequest.mockResolvedValue({ items: [], total: 0 })

    await listScheduledTaskDefinitions({
      limit: 25,
      offset: 10,
      family: "agent_task",
      lifecycle: "paused",
      q: "agent task"
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        method: "GET",
        path: "/api/v1/scheduled-tasks/definitions?limit=25&offset=10&family=agent_task&lifecycle=paused&q=agent+task"
      })
    )
  })

  it("normalizes projected automation ids for lifecycle definition routes", async () => {
    mocks.bgRequest.mockResolvedValue({ id: "def_1" })

    await pauseScheduledTaskDefinition("automation_definition:def_1", {
      idempotencyKey: "pause-key-1"
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        method: "POST",
        path: "/api/v1/scheduled-tasks/definitions/def_1/pause",
        headers: { "Idempotency-Key": "pause-key-1" }
      })
    )
  })

  it("normalizes raw and projected automation ids for detail and audit routes", async () => {
    mocks.bgRequest.mockResolvedValue({ id: "def_1" })

    await getScheduledTaskDefinition("def_1")
    await listScheduledTaskDefinitionAudit("automation_definition:def_1", {
      limit: 10,
      event_type: "paused"
    })

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        method: "GET",
        path: "/api/v1/scheduled-tasks/definitions/def_1"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        method: "GET",
        path: "/api/v1/scheduled-tasks/definitions/def_1/audit?limit=10&event_type=paused"
      })
    )
  })

  it("rejects wrong projected primitive prefixes before definition mutations", async () => {
    await expect(
      archiveScheduledTaskDefinition("watchlist_job:def_1")
    ).rejects.toThrow("Definition mutations require an automation_definition id")

    expect(mocks.bgRequest).not.toHaveBeenCalled()
  })
})
