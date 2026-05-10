import { describe, expect, it } from "vitest"

import { buildPersonaBuddyDiagnostics } from "../personaBuddyDiagnostics"

describe("buildPersonaBuddyDiagnostics", () => {
  it("returns healthy diagnostics for connected persona live state", () => {
    const diagnostics = buildPersonaBuddyDiagnostics({
      selectedPersona: { id: "persona-1", name: "Ada" },
      profileState: "loaded",
      buddySummary: "Uses the active persona profile.",
      capabilities: { hasPersona: true, hasMcp: true },
      liveSession: { connected: true, connecting: false, sessionId: "session-1" },
      liveVoice: { state: "idle", recoveryMode: "none" },
      wake: { armed: true, detectorState: "ready", triggerPhrases: ["hey ada"] },
      visual: { packLoadStatus: "loaded", diagnostic: null }
    })

    expect(diagnostics.state).toBe("healthy")
    expect(diagnostics.rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: "Persona", value: "Ada" }),
        expect.objectContaining({ label: "Live session", value: "Connected" })
      ])
    )
  })

  it("marks missing persona support unavailable", () => {
    const diagnostics = buildPersonaBuddyDiagnostics({
      selectedPersona: null,
      profileState: "idle",
      buddySummary: null,
      capabilities: { hasPersona: false, hasMcp: false },
      liveSession: { connected: false, connecting: false, sessionId: null },
      liveVoice: { state: "idle", recoveryMode: "none" },
      wake: { armed: false, detectorState: "idle" },
      visual: { packLoadStatus: "idle", diagnostic: null }
    })

    expect(diagnostics.state).toBe("unavailable")
    expect(diagnostics.message).toMatch(/persona/i)
  })

  it("marks reconnecting live voice state recovering", () => {
    const diagnostics = buildPersonaBuddyDiagnostics({
      selectedPersona: { id: "persona-1", name: "Ada" },
      profileState: "loaded",
      buddySummary: "Ready",
      capabilities: { hasPersona: true, hasMcp: true },
      liveSession: { connected: false, connecting: true, sessionId: "session-1" },
      liveVoice: { state: "listening", recoveryMode: "reconnect" },
      wake: { armed: true, detectorState: "ready" },
      visual: { packLoadStatus: "loaded", diagnostic: null }
    })

    expect(diagnostics.state).toBe("recovering")
    expect(diagnostics.rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: "Live session", state: "recovering" })
      ])
    )
  })

  it("marks broken visual packs degraded without treating no active pack as broken", () => {
    const degraded = buildPersonaBuddyDiagnostics({
      selectedPersona: { id: "persona-1", name: "Ada" },
      profileState: "loaded",
      buddySummary: "Ready",
      capabilities: { hasPersona: true, hasMcp: true },
      liveSession: { connected: true, connecting: false, sessionId: "session-1" },
      liveVoice: { state: "idle", recoveryMode: "none" },
      wake: { armed: true, detectorState: "ready" },
      visual: {
        packLoadStatus: "error",
        diagnostic: {
          code: "missing_manifest",
          severity: "warning",
          title: "Visual pack manifest is missing",
          message: "Visual pack manifest is missing."
        }
      }
    })

    expect(degraded.state).toBe("degraded")
    expect(degraded.rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: "Visual pack", state: "degraded" })
      ])
    )

    const fallback = buildPersonaBuddyDiagnostics({
      selectedPersona: { id: "persona-1", name: "Ada" },
      profileState: "loaded",
      buddySummary: "Ready",
      capabilities: { hasPersona: true, hasMcp: true },
      liveSession: { connected: true, connecting: false, sessionId: "session-1" },
      liveVoice: { state: "idle", recoveryMode: "none" },
      wake: { armed: true, detectorState: "ready" },
      visual: {
        packLoadStatus: "idle",
        diagnostic: {
          code: "no_active_pack",
          severity: "info",
          title: "No active visual pack",
          message: "This persona is using the text Buddy fallback."
        }
      }
    })

    expect(fallback.state).toBe("healthy")
    expect(fallback.rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: "Visual pack", state: "healthy" })
      ])
    )
  })

  it("surfaces wake warnings and MCP readiness as degraded diagnostics", () => {
    const diagnostics = buildPersonaBuddyDiagnostics({
      selectedPersona: { id: "persona-1", name: "Ada" },
      profileState: "loaded",
      buddySummary: "Ready",
      capabilities: { hasPersona: true, hasMcp: false },
      liveSession: { connected: true, connecting: false, sessionId: "session-1" },
      liveVoice: { state: "idle", recoveryMode: "none" },
      wake: {
        armed: true,
        detectorState: "unavailable",
        warning: "Wake listening is unavailable in this browser context."
      },
      visual: { packLoadStatus: "loaded", diagnostic: null }
    })

    expect(diagnostics.state).toBe("degraded")
    expect(diagnostics.rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: "Wake", state: "degraded" }),
        expect.objectContaining({ label: "MCP persona_visuals", state: "degraded" })
      ])
    )
  })
})
