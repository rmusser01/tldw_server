import { renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const client = vi.hoisted(() => ({ getChat: vi.fn() }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: client }))

import { useForkParentChat } from "../useForkParentChat"
import { getQueryClient } from "@/services/query-client"

describe("useForkParentChat (CM-12)", () => {
  beforeEach(() => {
    client.getChat.mockReset()
    getQueryClient().clear()
  })

  it("finds the chat a branch was forked from", async () => {
    client.getChat.mockImplementation(async (id: string) =>
      id === "branch"
        ? { id: "branch", title: "Trip plan — branch at message 3", parent_conversation_id: "trip" }
        : { id: "trip", title: "Trip plan" }
    )
    const { result } = renderHook(() => useForkParentChat("branch"))
    await waitFor(() => expect(result.current).toEqual({ id: "trip", title: "Trip plan" }))
    expect(client.getChat).toHaveBeenCalledWith("branch", expect.objectContaining({ signal: expect.any(AbortSignal) }))
  })

  it("has no parent for a chat that isn't a branch, and asks nothing for no chat", async () => {
    client.getChat.mockResolvedValue({ id: "plain", title: "Plain", parent_conversation_id: null })
    const { result, rerender } = renderHook(({ id }) => useForkParentChat(id), {
      initialProps: { id: "plain" as string | null }
    })
    await waitFor(() => expect(client.getChat).toHaveBeenCalledOnce())
    expect(result.current).toBeNull()
    rerender({ id: null })
    expect(result.current).toBeNull()
    expect(client.getChat).toHaveBeenCalledOnce()
  })

  it("never shows a previous chat's parent for the chat now open", async () => {
    let releaseOld!: (value: unknown) => void
    client.getChat.mockImplementation((id: string) =>
      id === "old-branch"
        ? new Promise((resolve) => { releaseOld = resolve })
        : Promise.resolve({ id, title: "Other", parent_conversation_id: null })
    )
    const { result, rerender } = renderHook(({ id }) => useForkParentChat(id), {
      initialProps: { id: "old-branch" }
    })
    rerender({ id: "other" })
    releaseOld({ id: "old-branch", parent_conversation_id: "old-parent" })
    await waitFor(() => expect(client.getChat).toHaveBeenCalledWith("other", expect.anything()))
    expect(result.current).toBeNull()
    expect(client.getChat).not.toHaveBeenCalledWith("old-parent", expect.anything())
  })

  it("leaves the lineage out when the parent can't be read", async () => {
    client.getChat.mockImplementation(async (id: string) => {
      if (id === "branch") return { id: "branch", parent_conversation_id: "deleted" }
      throw new Error("HTTP 404")
    })
    const { result } = renderHook(() => useForkParentChat("branch"))
    await waitFor(() => expect(client.getChat).toHaveBeenCalledTimes(2))
    expect(result.current).toBeNull()
  })
})

describe("useForkParentChat reads (review follow-up)", () => {
  beforeEach(() => {
    client.getChat.mockReset()
    getQueryClient().clear()
  })

  it("reads a branch and its parent's title once, however often the chat is reopened", async () => {
    client.getChat.mockImplementation(async (id: string) =>
      id === "branch" ? { id: "branch", parent_conversation_id: "trip" } : { id: "trip", title: "Trip plan" })
    const first = renderHook(() => useForkParentChat("branch"))
    await waitFor(() => expect(first.result.current).toEqual({ id: "trip", title: "Trip plan" }))
    first.unmount()
    const second = renderHook(() => useForkParentChat("branch"))
    await waitFor(() => expect(second.result.current).toEqual({ id: "trip", title: "Trip plan" }))
    expect(client.getChat).toHaveBeenCalledTimes(2)
  })

  it("takes the parent from the chat list the sidebar already read, fetching only the parent's title", async () => {
    getQueryClient().setQueryData(["serverChatHistory", { page: 1 }], {
      items: [{ id: "branch", title: "Trip plan — branch at message 3", parent_conversation_id: "trip" }],
      total: 1
    })
    client.getChat.mockResolvedValue({ id: "trip", title: "Trip plan" })
    const { result } = renderHook(() => useForkParentChat("branch"))
    await waitFor(() => expect(result.current).toEqual({ id: "trip", title: "Trip plan" }))
    expect(client.getChat).toHaveBeenCalledOnce()
    expect(client.getChat).toHaveBeenCalledWith("trip", expect.anything())
  })
})
