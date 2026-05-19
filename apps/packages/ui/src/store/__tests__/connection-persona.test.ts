import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ConnectionPhase } from "@/types/connection"
import type { UserPersona } from "@/types/connection"

vi.mock("@/services/tldw-server", () => ({
  getStoredTldwServerURL: vi.fn(async () => null)
}))

vi.mock("@/services/api-send", () => ({
  apiSend: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: vi.fn(),
    initialize: vi.fn(),
    ragHealth: vi.fn(),
    updateConfig: vi.fn()
  }
}))

import { useConnectionStore } from "../connection"

const setConnectionState = (overrides: Record<string, unknown>) => {
  const prev = useConnectionStore.getState().state
  useConnectionStore.setState({
    state: {
      ...prev,
      ...overrides
    }
  })
}

describe("UserPersona type and persistence", () => {
  beforeEach(async () => {
    localStorage.clear()
    await useConnectionStore.getState().setUserPersona(null)
  })

  afterEach(async () => {
    localStorage.clear()
    await useConnectionStore.getState().setUserPersona(null)
  })

  it("defaults to null in initial state", () => {
    const state = useConnectionStore.getState().state
    expect(state.userPersona).toBeNull()
  })

  it("defaults to null in localStorage", () => {
    const stored = localStorage.getItem("__tldw_user_persona")
    expect(stored).toBeNull()
  })

  it("accepts valid persona values", async () => {
    const validPersonas: UserPersona[] = ["family", "researcher", "explorer", null]
    for (const persona of validPersonas) {
      await useConnectionStore.getState().setUserPersona(persona)
      expect(useConnectionStore.getState().state.userPersona).toBe(persona)
      if (persona) {
        expect(localStorage.getItem("__tldw_user_persona")).toBe(persona)
      } else {
        expect(localStorage.getItem("__tldw_user_persona")).toBeNull()
      }
    }
  })

  it("setUserPersona persists to localStorage and updates state", async () => {
    await useConnectionStore.getState().setUserPersona("researcher")
    expect(useConnectionStore.getState().state.userPersona).toBe("researcher")
    expect(localStorage.getItem("__tldw_user_persona")).toBe("researcher")
  })

  it("setUserPersona(null) removes localStorage key and clears state", async () => {
    await useConnectionStore.getState().setUserPersona("family")
    expect(localStorage.getItem("__tldw_user_persona")).toBe("family")

    await useConnectionStore.getState().setUserPersona(null)
    expect(useConnectionStore.getState().state.userPersona).toBeNull()
    expect(localStorage.getItem("__tldw_user_persona")).toBeNull()
  })

  it("cycles through all persona values", async () => {
    const personas: UserPersona[] = ["family", "researcher", "explorer", null]
    for (const persona of personas) {
      await useConnectionStore.getState().setUserPersona(persona)
      expect(useConnectionStore.getState().state.userPersona).toBe(persona)
    }
  })

  it("does not clobber other state fields when setting persona", async () => {
    setConnectionState({
      phase: ConnectionPhase.CONNECTED,
      isConnected: true,
      hasCompletedFirstRun: true,
      serverUrl: "http://localhost:8000"
    })

    await useConnectionStore.getState().setUserPersona("explorer")

    const state = useConnectionStore.getState().state
    expect(state.userPersona).toBe("explorer")
    expect(state.phase).toBe(ConnectionPhase.CONNECTED)
    expect(state.isConnected).toBe(true)
    expect(state.hasCompletedFirstRun).toBe(true)
    expect(state.serverUrl).toBe("http://localhost:8000")
  })
})
