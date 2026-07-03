import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { Storage } from "@web/extension/shims/plasmo-storage"

describe("plasmo storage web shim", () => {
  beforeEach(() => {
    localStorage.clear()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("keeps local and sync areas from clobbering each other", async () => {
    const local = new Storage({ area: "local" })
    const sync = new Storage({ area: "sync" })
    const assistant = {
      kind: "character",
      id: "char-mira",
      name: "Mira"
    }

    await local.set("selectedAssistant", assistant)
    await sync.remove("selectedAssistant")

    expect(await local.get("selectedAssistant")).toEqual(assistant)
    expect(await sync.get("selectedAssistant")).toBeUndefined()
  })

  it("does not throw when a foreign tab writes a non-JSON value to a watched key", () => {
    const watcher = new Storage({ area: "local" })
    const changes: Array<{ newValue: unknown }> = []
    const unwatch = watcher.watch({
      fromAnotherTab: (change) => changes.push({ newValue: change.newValue })
    })

    // Simulate the browser `storage` event that fires in *other* tabs, carrying
    // a value that is NOT valid JSON. The window handler must not explode.
    expect(() =>
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: "fromAnotherTab",
          oldValue: null,
          newValue: "definitely-not-json{{{"
        })
      )
    ).not.toThrow()

    // The watcher still fires and receives the raw (undeserializable) string.
    expect(changes).toEqual([{ newValue: "definitely-not-json{{{" }])
    unwatch()
  })

  it("logs (does not silently swallow) a throwing watch callback and keeps siblings working", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {})
    const writer = new Storage({ area: "local" })
    const watcher = new Storage({ area: "local" })

    const siblingCalls: unknown[] = []
    const unwatchBoom = watcher.watch({
      boom: () => {
        throw new Error("watcher kaboom")
      }
    })
    const unwatchSibling = watcher.watch({
      boom: (change) => siblingCalls.push(change.newValue)
    })

    await writer.set("boom", 42)

    // A sibling watcher still runs despite the first callback throwing ...
    expect(siblingCalls).toEqual([42])
    // ... and the failure is surfaced via console.error, not swallowed.
    expect(errorSpy).toHaveBeenCalled()

    unwatchBoom()
    unwatchSibling()
  })
})
