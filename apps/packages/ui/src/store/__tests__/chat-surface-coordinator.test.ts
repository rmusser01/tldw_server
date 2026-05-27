// @vitest-environment jsdom
import { describe, expect, it } from "vitest"

import {
  createChatSurfaceCoordinatorStore,
  shouldEnableOptionalResource
} from "@/store/chat-surface-coordinator"

describe("chat-surface-coordinator", () => {
  it("tracks optional panels only after they are visible and engaged", () => {
    const store = createChatSurfaceCoordinatorStore()

    expect(store.getState().visiblePanels).toHaveProperty("model-catalog", false)
    expect(store.getState().engagedPanels).toHaveProperty("model-catalog", false)

    store.getState().setPanelVisible("model-catalog", true)

    expect(
      shouldEnableOptionalResource(store.getState(), "model-catalog")
    ).toBe(false)

    store.getState().markPanelEngaged("model-catalog")

    expect(
      shouldEnableOptionalResource(store.getState(), "model-catalog")
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
