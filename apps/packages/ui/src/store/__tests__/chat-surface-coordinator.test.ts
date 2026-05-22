// @vitest-environment jsdom
import { describe, expect, it } from "vitest"

import {
  createChatSurfaceCoordinatorStore,
  shouldEnableOptionalResource
} from "@/store/chat-surface-coordinator"

describe("chat-surface-coordinator", () => {
  it("tracks the character control panel alongside the existing optional panels", () => {
    const store = createChatSurfaceCoordinatorStore()

    expect(store.getState().visiblePanels).toHaveProperty("character-control", false)
    expect(store.getState().engagedPanels).toHaveProperty("character-control", false)

    store.getState().setPanelVisible("character-control", true)

    expect(
      shouldEnableOptionalResource(store.getState(), "character-control")
    ).toBe(false)

    store.getState().markPanelEngaged("character-control")

    expect(
      shouldEnableOptionalResource(store.getState(), "character-control")
    ).toBe(true)
  })

  it("keeps server history disabled until the user engages the panel", () => {
    const store = createChatSurfaceCoordinatorStore()

    store.getState().setRouteContext({ routeId: "chat", surface: "webui" })
    store.getState().setPanelVisible("server-history", true)

    expect(
      shouldEnableOptionalResource(store.getState(), "server-history")
    ).toBe(false)

    store.getState().markPanelEngaged("server-history")

    expect(
      shouldEnableOptionalResource(store.getState(), "server-history")
    ).toBe(true)
  })

  it("does not enable a panel that was engaged before it becomes visible", () => {
    const store = createChatSurfaceCoordinatorStore()

    store.getState().markPanelEngaged("mcp-tools")

    expect(shouldEnableOptionalResource(store.getState(), "mcp-tools")).toBe(
      false
    )

    store.getState().setPanelVisible("mcp-tools", true)

    expect(shouldEnableOptionalResource(store.getState(), "mcp-tools")).toBe(
      true
    )
  })
})
