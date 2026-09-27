import { beforeEach, describe, expect, it, vi } from "vitest"

const loadStoreModule = async () =>
  import("../persona-buddy-shell")

describe("persona buddy shell store", () => {
  beforeEach(() => {
    localStorage.clear()
    vi.resetModules()
  })

  it("defaults to a compact closed shell with per-bucket positions", async () => {
    const module = await loadStoreModule()
    const state = module.usePersonaBuddyShellStore.getState()

    expect(state.isOpen).toBe(false)
    expect(state.getPosition("web-desktop")).toEqual(
      module.DEFAULT_PERSONA_BUDDY_SHELL_POSITIONS["web-desktop"]
    )
    expect(state.getPosition("sidepanel-desktop")).toEqual(
      module.DEFAULT_PERSONA_BUDDY_SHELL_POSITIONS["sidepanel-desktop"]
    )
  })

  it.each(["web-desktop", "sidepanel-desktop"] as const)(
    "keeps fresh and reset %s placement below navigation across desktop and compact viewports",
    async (bucket) => {
      const storeModule = await loadStoreModule()
      const state = storeModule.usePersonaBuddyShellStore.getState()
      const otherBucket =
        bucket === "web-desktop" ? "sidepanel-desktop" : "web-desktop"
      state.setPosition(otherBucket, { x: 28, y: 440 })

      for (const { expectedX, ...bounds } of [
        { viewportWidth: 1090, viewportHeight: 990, expectedX: 942 },
        { viewportWidth: 390, viewportHeight: 844, expectedX: 242 }
      ]) {
        const shellBounds = { ...bounds, shellWidth: 132, shellHeight: 174 }
        expect(
          storeModule.clampPersonaBuddyShellPosition(
            state.getPosition(bucket),
            bucket,
            shellBounds
          )
        ).toEqual({ x: expectedX, y: 96 })

        state.setPosition(bucket, { x: 360, y: 640 })
        state.resetPosition(bucket)

        expect(
          storeModule.clampPersonaBuddyShellPosition(
            state.getPosition(bucket),
            bucket,
            shellBounds
          )
        ).toEqual({ x: expectedX, y: 96 })
        expect(state.getPosition(otherBucket)).toEqual({ x: 28, y: 440 })
      }
    }
  )

  it("rehydrates position memory without persisting the open shell session", async () => {
    const module = await loadStoreModule()
    const state = module.usePersonaBuddyShellStore.getState()

    state.setOpen(true)
    state.setPosition("web-desktop", { x: 420, y: 168 })
    state.setPosition("sidepanel-desktop", { x: 1120, y: 640 })

    expect(localStorage.getItem(module.PERSONA_BUDDY_SHELL_STORAGE_KEY)).toBeTruthy()

    vi.resetModules()
    const reloadedModule = await loadStoreModule()
    await reloadedModule.usePersonaBuddyShellStore.persist.rehydrate()

    const reloadedState = reloadedModule.usePersonaBuddyShellStore.getState()
    expect(reloadedState.isOpen).toBe(false)
    expect(reloadedState.getPosition("web-desktop")).toEqual({ x: 420, y: 168 })
    expect(reloadedState.getPosition("sidepanel-desktop")).toEqual({
      x: 1120,
      y: 640
    })
  })

  it("keeps position memory separate for web and sidepanel desktop buckets", async () => {
    const module = await loadStoreModule()
    const state = module.usePersonaBuddyShellStore.getState()

    state.setPosition("web-desktop", { x: 360, y: 144 })
    state.setPosition("sidepanel-desktop", { x: 28, y: 88 })

    expect(state.getPosition("web-desktop")).toEqual({ x: 360, y: 144 })
    expect(state.getPosition("sidepanel-desktop")).toEqual({ x: 28, y: 88 })
  })

  it("resolves, resets, and clamps missing bucket positions safely", async () => {
    const module = await loadStoreModule()

    expect(
      module.resolvePersonaBuddyShellPosition({}, "web-desktop")
    ).toEqual(module.DEFAULT_PERSONA_BUDDY_SHELL_POSITIONS["web-desktop"])

    expect(
      module.resetPersonaBuddyShellPositionBucket({}, "sidepanel-desktop")
    ).toEqual({
      "web-desktop": module.DEFAULT_PERSONA_BUDDY_SHELL_POSITIONS["web-desktop"],
      "sidepanel-desktop":
        module.DEFAULT_PERSONA_BUDDY_SHELL_POSITIONS["sidepanel-desktop"]
    })

    expect(
      module.clampPersonaBuddyShellPosition(undefined, "web-desktop", {
        viewportWidth: 120,
        viewportHeight: 120,
        shellWidth: 96,
        shellHeight: 96,
        margin: 12
      })
    ).toEqual({ x: 12, y: 12 })
  })
})
