import { QueryClient } from "@tanstack/react-query"
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { patchSavedNoteInListCache } from "../notes-list-cache"
import {
  NOTES_LIST_REFRESH_DELAY_MS,
  useNotesListSaveRefresh
} from "../hooks/useNotesListSaveRefresh"
import type { NoteListItem } from "../notes-manager-types"

// NS-04: a save patches its row in the cached list and refreshes the list in
// the background, once, instead of awaiting a full reload.

const rows = (): NoteListItem[] => [
  { id: "n1", title: "Old title", content: "Old body", updated_at: "2026-10-01T10:00:00.000Z", version: 1, keywords: [] },
  { id: "n2", title: "Other", content: "Other body", updated_at: "2026-09-30T10:00:00.000Z", version: 4, keywords: ["x"] }
]

describe("patchSavedNoteInListCache (NS-04)", () => {
  it("updates the saved row in every cached list and leaves other rows alone", () => {
    const queryClient = new QueryClient()
    queryClient.setQueryData(["notes", "active", "list", 1], rows())
    queryClient.setQueryData(["notes", "active", "list", 2], [rows()[1]])
    queryClient.setQueryData(["notes-moodboards"], [{ id: "n1", title: "Unrelated" }])

    const patched = patchSavedNoteInListCache(queryClient, "n1", {
      title: "New title",
      content: "New body",
      version: 2,
      last_modified: "2026-10-02T10:00:00.000Z",
      keywords: [{ keyword: "research" }]
    })

    expect(patched).toBe(1)
    const pageOne = queryClient.getQueryData<NoteListItem[]>(["notes", "active", "list", 1])
    expect(pageOne?.[0]).toMatchObject({
      id: "n1",
      title: "New title",
      content: "New body",
      version: 2,
      updated_at: "2026-10-02T10:00:00.000Z",
      keywords: ["research"]
    })
    expect(pageOne?.[1]).toEqual(rows()[1])
    expect(queryClient.getQueryData(["notes-moodboards"])).toEqual([{ id: "n1", title: "Unrelated" }])
  })

  it("falls back to the submitted title and tags when the response omits them", () => {
    const queryClient = new QueryClient()
    queryClient.setQueryData(["notes", "active"], rows())

    patchSavedNoteInListCache(
      queryClient,
      "n1",
      { version: 3 },
      { title: "Typed title", keywords: ["draft"] }
    )

    expect(queryClient.getQueryData<NoteListItem[]>(["notes", "active"])?.[0]).toMatchObject({
      title: "Typed title",
      version: 3,
      keywords: ["draft"],
      updated_at: "2026-10-01T10:00:00.000Z"
    })
  })

  it("leaves lists cached for another server or account alone, even when note ids match", () => {
    const queryClient = new QueryClient()
    const thisServer = ["notes", "active", "authority", "server-a::user-1", "generation", 1]
    const otherServer = ["notes", "active", "authority", "server-b::user-1", "generation", 1]
    queryClient.setQueryData(thisServer, rows())
    queryClient.setQueryData(otherServer, rows())

    const patched = patchSavedNoteInListCache(
      queryClient,
      "n1",
      { title: "Saved on server A" },
      {},
      { authorityScope: "server-a::user-1" }
    )

    expect(patched).toBe(1)
    expect(queryClient.getQueryData<NoteListItem[]>(thisServer)?.[0].title).toBe("Saved on server A")
    expect(queryClient.getQueryData<NoteListItem[]>(otherServer)?.[0].title).toBe("Old title")
  })
})

describe("useNotesListSaveRefresh (NS-04)", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("patches immediately and runs one background refresh for saves close together", async () => {
    const queryClient = new QueryClient()
    queryClient.setQueryData(["notes", "active"], rows())
    const refetch = vi.fn(async () => undefined)
    const { result } = renderHook(() => useNotesListSaveRefresh(queryClient, refetch))

    act(() => {
      result.current("n1", { title: "First save" })
    })
    expect(queryClient.getQueryData<NoteListItem[]>(["notes", "active"])?.[0].title).toBe("First save")
    expect(refetch).not.toHaveBeenCalled()

    await act(async () => {
      vi.advanceTimersByTime(NOTES_LIST_REFRESH_DELAY_MS - 1)
    })
    act(() => {
      result.current("n1", { title: "Second save" })
    })
    await act(async () => {
      vi.advanceTimersByTime(NOTES_LIST_REFRESH_DELAY_MS)
    })

    expect(refetch).toHaveBeenCalledTimes(1)
    expect(queryClient.getQueryData<NoteListItem[]>(["notes", "active"])?.[0].title).toBe("Second save")
  })

  it("drops a pending refresh when the page unmounts", async () => {
    const queryClient = new QueryClient()
    const refetch = vi.fn(async () => undefined)
    const { result, unmount } = renderHook(() => useNotesListSaveRefresh(queryClient, refetch))

    act(() => {
      result.current("n1", { title: "Saved" })
    })
    unmount()
    await act(async () => {
      vi.advanceTimersByTime(NOTES_LIST_REFRESH_DELAY_MS * 2)
    })

    expect(refetch).not.toHaveBeenCalled()
  })
})
