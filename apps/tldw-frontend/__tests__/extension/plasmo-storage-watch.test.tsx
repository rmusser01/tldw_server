import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { Storage } from "@web/extension/shims/plasmo-storage"
import { useStorage } from "@web/extension/shims/plasmo-storage-hook"

describe("plasmo storage cross-instance change propagation (H10)", () => {
  beforeEach(() => {
    localStorage.clear()
  })

  afterEach(() => {
    cleanup()
  })

  it("notifies a watcher on a different instance when another instance writes", async () => {
    const writer = new Storage({ area: "local" })
    const watcher = new Storage({ area: "local" })

    const changes: unknown[] = []
    const unwatch = watcher.watch({
      stickyChatInput: (change) => changes.push(change.newValue)
    })

    await writer.set("stickyChatInput", true)

    expect(changes).toEqual([true])
    unwatch()
  })

  it("keeps areas isolated: a sync write does not notify a local watcher", async () => {
    const localWatcher = new Storage({ area: "local" })
    const syncWriter = new Storage({ area: "sync" })

    const localChanges: unknown[] = []
    const unwatch = localWatcher.watch({
      shared: (change) => localChanges.push(change.newValue)
    })

    await syncWriter.set("shared", "sync-only")

    expect(localChanges).toEqual([])
    unwatch()
  })

  it("useStorage reflects a value written by another instance without a reload", async () => {
    const external = new Storage({ area: "local" })

    const { result } = renderHook(() =>
      useStorage<boolean>("stickyChatInput", false)
    )

    // initial default value once loading settles
    await waitFor(() => expect(result.current[2].isLoading).toBe(false))
    expect(result.current[0]).toBe(false)

    // a write from a *different* Storage instance should propagate
    await act(async () => {
      await external.set("stickyChatInput", true)
    })

    await waitFor(() => expect(result.current[0]).toBe(true))
  })

  it("does not clobber a fresher watch() update with a stale initial get()", async () => {
    const instance = new Storage({ area: "local" })

    // Control exactly when the initial get() resolves so we can interleave a
    // watch update in between the synchronous get() call and its microtask.
    let resolveGet: (value: string | undefined) => void = () => {}
    vi.spyOn(instance, "get").mockImplementation(
      () =>
        new Promise<string | undefined>((resolve) => {
          resolveGet = resolve
        }) as Promise<never>
    )

    const external = new Storage({ area: "local" })

    const { result } = renderHook(() =>
      useStorage<string>({ key: "raced", instance, defaultValue: "default" })
    )

    // A fresher value arrives via watch() before the initial get() resolves.
    await act(async () => {
      await external.set("raced", "fresh")
    })
    expect(result.current[0]).toBe("fresh")

    // The stale initial get() now resolves — it must NOT overwrite "fresh".
    await act(async () => {
      resolveGet("stale")
      await Promise.resolve()
    })

    expect(result.current[0]).toBe("fresh")
    vi.restoreAllMocks()
  })

  it("functional setValue uses the freshest value, not a stale closure", async () => {
    const { result } = renderHook(() => useStorage<number>("counter", 0))

    await waitFor(() => expect(result.current[2].isLoading).toBe(false))

    // Two functional updates in a row must compound (0 -> 1 -> 2), not drop.
    await act(async () => {
      await result.current[1]((prev) => (prev ?? 0) + 1)
    })
    await act(async () => {
      await result.current[1]((prev) => (prev ?? 0) + 1)
    })

    expect(result.current[0]).toBe(2)
  })
})
