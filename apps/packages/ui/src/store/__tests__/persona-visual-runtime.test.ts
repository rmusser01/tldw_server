import { beforeEach, describe, expect, it } from "vitest"

import { usePersonaVisualRuntimeStore } from "../persona-visual-runtime"

describe("persona visual runtime store", () => {
  beforeEach(() => {
    usePersonaVisualRuntimeStore.setState({
      override: null,
      runtimeDiagnostics: null
    })
  })

  it("stores and clears expired runtime overrides", () => {
    usePersonaVisualRuntimeStore.getState().setOverride({
      personaId: "persona-1",
      sessionId: "session-1",
      state: "speaking",
      reason: "mcp-trigger",
      expiresAt: 2000
    })

    expect(usePersonaVisualRuntimeStore.getState().override?.state).toBe("speaking")

    usePersonaVisualRuntimeStore.getState().clearExpired(1999)
    expect(usePersonaVisualRuntimeStore.getState().override?.state).toBe("speaking")

    usePersonaVisualRuntimeStore.getState().clearExpired(2000)
    expect(usePersonaVisualRuntimeStore.getState().override).toBeNull()
  })

  it("clears overrides for the matching live session only", () => {
    usePersonaVisualRuntimeStore.getState().setOverride({
      personaId: "persona-1",
      sessionId: "session-1",
      state: "error",
      reason: "demo",
      expiresAt: 2000
    })

    usePersonaVisualRuntimeStore.getState().clearForSession("other-session")
    expect(usePersonaVisualRuntimeStore.getState().override?.state).toBe("error")

    usePersonaVisualRuntimeStore.getState().clearForSession("session-1")
    expect(usePersonaVisualRuntimeStore.getState().override).toBeNull()
  })

  it("clears runtime diagnostics only for the matching source", () => {
    usePersonaVisualRuntimeStore.getState().setRuntimeDiagnostics({
      sourceId: "sidepanel:persona-garden",
      personaId: "persona-1",
      sessionId: null,
      packId: "pack-1",
      packTitle: "Animated buddy",
      packLoadStatus: "loaded",
      visualState: "idle",
      diagnostic: null,
      updatedAt: 2000
    })

    usePersonaVisualRuntimeStore
      .getState()
      .clearRuntimeDiagnostics("web:persona-garden")
    expect(usePersonaVisualRuntimeStore.getState().runtimeDiagnostics?.packId).toBe(
      "pack-1"
    )

    usePersonaVisualRuntimeStore
      .getState()
      .clearRuntimeDiagnostics("sidepanel:persona-garden")
    expect(usePersonaVisualRuntimeStore.getState().runtimeDiagnostics).toBeNull()
  })
})
