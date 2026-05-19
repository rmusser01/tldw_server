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

  it("rehydrates position memory without persisting the open shell session", async () => {
    const module = await loadStoreModule()
    const state = module.usePersonaBuddyShellStore.getState()

    state.setOpen(true)
    state.setPosition("web-desktop", { x: 420, y: 168 })

    expect(localStorage.getItem(module.PERSONA_BUDDY_SHELL_STORAGE_KEY)).toBeTruthy()

    vi.resetModules()
    const reloadedModule = await loadStoreModule()
    await reloadedModule.usePersonaBuddyShellStore.persist.rehydrate()

    const reloadedState = reloadedModule.usePersonaBuddyShellStore.getState()
    expect(reloadedState.isOpen).toBe(false)
    expect(reloadedState.getPosition("web-desktop")).toEqual({ x: 420, y: 168 })
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
