import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useRecipePersistenceOwner } from "../useRecipePersistenceOwner"

const resolveOwner = vi.hoisted(() => vi.fn())
vi.mock("@/services/recipe-persistence-uncertainty", () => ({
  resolveRecipePersistenceOwnerView: resolveOwner
}))

const first = { ownerId: "owner-a", authorizationRevision: "revision-a" }
const rotated = { ownerId: "owner-a", authorizationRevision: "revision-b" }
function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

describe("useRecipePersistenceOwner", () => {
  beforeEach(() => resolveOwner.mockReset())

  it("does not resolve while disabled, even on explicit refresh", async () => {
    const { result } = renderHook(() => useRecipePersistenceOwner(false))
    await act(async () => {
      expect(await result.current.refresh()).toBeNull()
    })
    expect(resolveOwner).not.toHaveBeenCalled()
    expect(result.current.owner).toBeNull()
    expect(result.current.loading).toBe(false)
  })

  it("masks persistence on open and discards a late result after close", async () => {
    const pending = deferred<typeof first>()
    resolveOwner.mockReturnValueOnce(pending.promise)
    const { result, rerender } = renderHook(
      ({ enabled }) => useRecipePersistenceOwner(enabled),
      { initialProps: { enabled: false } }
    )
    rerender({ enabled: true })
    expect(result.current.owner).toBeNull()
    expect(result.current.loading).toBe(true)
    rerender({ enabled: false })
    await act(async () => {
      pending.resolve(first)
      await pending.promise
    })
    expect(result.current.owner).toBeNull()
    expect(result.current.loading).toBe(false)
  })

  it("refresh masks the old view and replaces a same-owner authorization revision", async () => {
    resolveOwner.mockResolvedValueOnce(first)
    const { result } = renderHook(() => useRecipePersistenceOwner(true))
    await waitFor(() => expect(result.current.owner).toEqual(first))
    const pending = deferred<typeof first>()
    resolveOwner.mockReturnValueOnce(pending.promise)
    let refresh!: Promise<typeof first | null>
    act(() => {
      refresh = result.current.refresh()
    })
    expect(result.current.owner).toBeNull()
    expect(result.current.loading).toBe(true)
    await act(async () => {
      pending.resolve(rotated)
      await refresh
    })
    expect(result.current.owner).toEqual(rotated)
    expect(result.current.loading).toBe(false)
  })

  it("reopens with a fresh owner and ignores the previous open's pending resolution", async () => {
    const pending = deferred<typeof first>()
    const second = { ownerId: "owner-b", authorizationRevision: "revision-c" }
    resolveOwner
      .mockReturnValueOnce(pending.promise)
      .mockResolvedValueOnce(second)
    const { result, rerender } = renderHook(
      ({ enabled }) => useRecipePersistenceOwner(enabled),
      { initialProps: { enabled: true } }
    )
    rerender({ enabled: false })
    rerender({ enabled: true })
    await waitFor(() => expect(result.current.owner).toEqual(second))
    await act(async () => {
      pending.resolve(first)
      await pending.promise
    })
    expect(result.current.owner).toEqual(second)
  })

  it.each([null, new Error("extension authority unavailable")])(
    "fails closed for missing or failed authority: %s",
    async (response) => {
      resolveOwner.mockResolvedValueOnce(first)
      const { result } = renderHook(() => useRecipePersistenceOwner(true))
      await waitFor(() => expect(result.current.owner).toEqual(first))
      if (response instanceof Error)
        resolveOwner.mockRejectedValueOnce(response)
      else resolveOwner.mockResolvedValueOnce(response)
      await act(async () => {
        await result.current.refresh()
      })
      expect(result.current.owner).toBeNull()
      expect(result.current.loading).toBe(false)
    }
  )
})
