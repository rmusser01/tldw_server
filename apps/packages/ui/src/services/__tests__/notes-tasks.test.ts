import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args)
}))

import {
  createNoteTask,
  deleteNoteTask,
  getTask,
  listNoteTasks,
  listTaskActivity,
  listTasks,
  markTaskActivityRead,
  reconcileNoteTasks,
  setNoteTaskStatus,
  updateNoteTask
} from "@/services/notes-tasks"

describe("notes task API helpers", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.bgRequest.mockResolvedValue({})
  })

  it("builds note-scoped list, create, and reconcile requests", async () => {
    await listNoteTasks("note/1", { limit: 25 })
    await createNoteTask("note/1", {
      text: "Draft PRD",
      status: "open",
      metadata: { priority: "high" },
      expected_note_version: 3
    })
    await reconcileNoteTasks("note/1")

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(1, {
      path: "/api/v1/notes/note%2F1/tasks?limit=25",
      method: "GET"
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(2, {
      path: "/api/v1/notes/note%2F1/tasks",
      method: "POST",
      body: {
        text: "Draft PRD",
        status: "open",
        metadata: { priority: "high" },
        expected_note_version: 3
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(3, {
      path: "/api/v1/notes/note%2F1/tasks/reconcile",
      method: "POST"
    })
  })

  it("builds global list, get, update, status, and delete requests", async () => {
    await listTasks({
      status: "open",
      projection_status: "live",
      limit: 50,
      reconcile_limit: 0
    })
    await getTask("task/1")
    await updateNoteTask("task/1", {
      text: "Updated task",
      metadata: { estimate: "2h" },
      expected_task_version: 4,
      expected_note_version: 7
    })
    await setNoteTaskStatus([
      {
        task_id: "task/1",
        status: "done",
        expected_task_version: 4,
        expected_note_version: 7
      }
    ])
    await deleteNoteTask("task/1", {
      expected_task_version: 5,
      expected_note_version: 8,
      record_only: true
    })

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(1, {
      path: "/api/v1/notes/tasks?status=open&projection_status=live&limit=50&reconcile_limit=0",
      method: "GET"
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(2, {
      path: "/api/v1/notes/tasks/task%2F1",
      method: "GET"
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(3, {
      path: "/api/v1/notes/tasks/task%2F1",
      method: "PATCH",
      body: {
        text: "Updated task",
        metadata: { estimate: "2h" },
        expected_task_version: 4,
        expected_note_version: 7
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(4, {
      path: "/api/v1/notes/tasks/status",
      method: "POST",
      body: {
        updates: [
          {
            task_id: "task/1",
            status: "done",
            expected_task_version: 4,
            expected_note_version: 7
          }
        ]
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(5, {
      path: "/api/v1/notes/tasks/task%2F1",
      method: "DELETE",
      body: {
        expected_task_version: 5,
        expected_note_version: 8,
        record_only: true
      }
    })
  })

  it("builds task activity list and read-state requests", async () => {
    await listTaskActivity({ limit: 10 })
    await markTaskActivityRead("event/1", { read: true })
    await markTaskActivityRead("event/2", { dismissed: true })

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(1, {
      path: "/api/v1/notes/tasks/activity?limit=10",
      method: "GET"
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(2, {
      path: "/api/v1/notes/tasks/activity/event%2F1",
      method: "PATCH",
      body: { read: true }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(3, {
      path: "/api/v1/notes/tasks/activity/event%2F2",
      method: "PATCH",
      body: { dismissed: true }
    })
  })
})
