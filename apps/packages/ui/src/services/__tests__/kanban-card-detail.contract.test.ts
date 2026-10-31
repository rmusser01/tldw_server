import { beforeEach, describe, expect, it, vi } from "vitest"

const transport = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({
  bgRequest: transport.request,
  bgUpload: vi.fn()
}))

import {
  createChecklist,
  createChecklistItem,
  listChecklists,
  listComments,
  updateChecklist,
  updateChecklistItem
} from "@/services/kanban"

// Canonical ChecklistResponse / ChecklistWithItemsResponse / CommentResponse.
const checklist = {
  id: 21,
  uuid: "checklist-21",
  card_id: 7,
  name: "Launch checks",
  position: 0,
  created_at: "2026-10-03T04:00:00Z",
  updated_at: "2026-10-03T04:00:00Z"
}
const item = {
  id: 31,
  uuid: "item-31",
  checklist_id: 21,
  name: "Confirm telescope alignment",
  checked: true,
  checked_at: "2026-10-03T04:01:00Z",
  position: 0,
  created_at: "2026-10-03T04:00:00Z",
  updated_at: "2026-10-03T04:01:00Z"
}
const detail = {
  ...checklist,
  items: [item],
  total_items: 1,
  checked_items: 1,
  progress_percent: 100
}
const comment = {
  id: 41,
  uuid: "comment-41",
  card_id: 7,
  user_id: "user-1",
  content: "Alignment confirmed.",
  created_at: "2026-10-03T04:02:00Z",
  updated_at: "2026-10-03T04:02:00Z",
  deleted: false
}

beforeEach(() => {
  transport.request.mockReset()
})

describe("Kanban card-detail API contract", () => {
  it("returns an iterable empty checklist list from the API envelope", async () => {
    transport.request.mockResolvedValue({ checklists: [] })
    const result = await listChecklists(7)
    expect(result.map((entry) => entry.id)).toEqual([])
    expect(transport.request).toHaveBeenCalledTimes(1)
  })

  it("loads checklist details in list order and exposes title, content and checked state", async () => {
    transport.request.mockImplementation(async ({ path }) => {
      if (path === "/api/v1/kanban/cards/7/checklists") {
        return {
          checklists: [checklist, { ...checklist, id: 22, position: 1 }]
        }
      }
      if (path === "/api/v1/kanban/checklists/21") return detail
      if (path === "/api/v1/kanban/checklists/22") {
        return {
          ...checklist,
          id: 22,
          name: "Empty checks",
          position: 1,
          items: [],
          total_items: 0,
          checked_items: 0,
          progress_percent: 0
        }
      }
      throw new Error(`Unexpected request: ${path}`)
    })
    const result = await listChecklists(7)
    expect(
      result.map((entry) => ({
        id: entry.id,
        title: entry.title,
        position: entry.position,
        items: entry.items.map((row) => ({
          id: row.id,
          content: row.content,
          checked: row.checked
        }))
      }))
    ).toEqual([
      {
        id: 21,
        title: "Launch checks",
        position: 0,
        items: [
          { id: 31, content: "Confirm telescope alignment", checked: true }
        ]
      },
      { id: 22, title: "Empty checks", position: 1, items: [] }
    ])
  })

  it("propagates a failed detail load instead of returning an empty checklist", async () => {
    transport.request
      .mockResolvedValueOnce({ checklists: [checklist] })
      .mockRejectedValueOnce(new Error("Checklist unavailable"))
    await expect(listChecklists(7)).rejects.toThrow("Checklist unavailable")
  })

  it("returns an empty comments array from the paginated envelope", async () => {
    transport.request.mockResolvedValue({
      comments: [],
      pagination: { total: 0, limit: 50, offset: 0, has_more: false }
    })
    expect(await listComments(7)).toEqual([])
  })

  it("retains comment identities and bodies from the paginated envelope", async () => {
    transport.request.mockResolvedValue({
      comments: [comment],
      pagination: { total: 1, limit: 50, offset: 0, has_more: false }
    })
    expect(await listComments(7)).toEqual([comment])
  })

  it("loads all comment pages in newest-first order", async () => {
    const comments = Array.from({ length: 151 }, (_, index) => ({
      ...comment,
      id: 151 - index,
      uuid: `comment-${151 - index}`,
      content: `Comment ${151 - index}`
    }))
    transport.request
      .mockResolvedValueOnce({
        comments: comments.slice(0, 100),
        pagination: { total: 151, limit: 100, offset: 0, has_more: true }
      })
      .mockResolvedValueOnce({
        comments: comments.slice(100),
        pagination: { total: 151, limit: 100, offset: 100, has_more: false }
      })

    expect(await listComments(7)).toEqual(comments)
    expect(
      transport.request.mock.calls.map(([request]) => request.path)
    ).toEqual([
      "/api/v1/kanban/cards/7/comments?limit=100&offset=0",
      "/api/v1/kanban/cards/7/comments?limit=100&offset=100"
    ])
  })

  it("propagates a later comment page failure instead of displaying an incomplete list", async () => {
    transport.request
      .mockResolvedValueOnce({
        comments: [comment],
        pagination: { total: 2, limit: 100, offset: 0, has_more: true }
      })
      .mockRejectedValueOnce(new Error("Comments unavailable"))

    await expect(listComments(7)).rejects.toThrow("Comments unavailable")
  })

  it("rejects a non-advancing comment page instead of requesting it forever", async () => {
    transport.request.mockResolvedValue({
      comments: [],
      pagination: { total: 1, limit: 100, offset: 0, has_more: true }
    })

    await expect(listComments(7)).rejects.toThrow(
      "Comment pagination did not advance"
    )
    expect(transport.request).toHaveBeenCalledTimes(1)
  })

  it("creates a checklist using API name and returns the UI title", async () => {
    transport.request.mockResolvedValue(checklist)
    const result = await createChecklist(7, {
      title: "Launch checks",
      client_id: "client-1"
    })
    expect(transport.request).toHaveBeenCalledWith({
      path: "/api/v1/kanban/cards/7/checklists",
      method: "POST",
      body: { name: "Launch checks", client_id: "client-1" }
    })
    expect(result).toMatchObject({ id: 21, card_id: 7, title: "Launch checks" })
  })

  it("renames a checklist using API name and returns its updated title", async () => {
    transport.request.mockResolvedValue({ ...checklist, name: "Ready checks" })
    const result = await updateChecklist(21, { title: "Ready checks" })
    expect(transport.request).toHaveBeenCalledWith({
      path: "/api/v1/kanban/checklists/21",
      method: "PATCH",
      body: { name: "Ready checks" }
    })
    expect(result).toMatchObject({ id: 21, title: "Ready checks" })
  })

  it("creates an item using API name and returns UI content and checked state", async () => {
    transport.request.mockResolvedValue(item)
    const result = await createChecklistItem(21, {
      content: "Confirm telescope alignment",
      client_id: "client-1"
    })
    expect(transport.request).toHaveBeenCalledWith({
      path: "/api/v1/kanban/checklists/21/items",
      method: "POST",
      body: { name: "Confirm telescope alignment", client_id: "client-1" }
    })
    expect(result).toMatchObject({
      id: 31,
      content: "Confirm telescope alignment",
      checked: true
    })
  })

  it("updates item content and false checked state using canonical fields", async () => {
    transport.request.mockResolvedValue({
      ...item,
      name: "Recheck alignment",
      checked: false
    })
    const result = await updateChecklistItem(31, {
      content: "Recheck alignment",
      checked: false
    })
    expect(transport.request).toHaveBeenCalledWith({
      path: "/api/v1/kanban/checklist-items/31",
      method: "PATCH",
      body: { name: "Recheck alignment", checked: false }
    })
    expect(result).toMatchObject({
      id: 31,
      content: "Recheck alignment",
      checked: false
    })
  })

  it("toggles an item without replacing its content", async () => {
    transport.request.mockResolvedValue({ ...item, checked: false })
    const result = await updateChecklistItem(31, { checked: false })
    expect(transport.request).toHaveBeenCalledWith({
      path: "/api/v1/kanban/checklist-items/31",
      method: "PATCH",
      body: { checked: false }
    })
    expect(result).toMatchObject({
      content: "Confirm telescope alignment",
      checked: false
    })
  })
})
