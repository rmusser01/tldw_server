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
      visual: {
        packId: "pack-1",
        packTitle: "Animated Pack",
        packLoadStatus: "loaded",
        visualState: "idle",
        diagnostic: null
      }
    })

    expect(diagnostics.state).toBe("healthy")
    expect(diagnostics.rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: "Persona", value: "Ada" }),
        expect.objectContaining({
          label: "Visual pack",
          value: "Animated Pack (pack-1)",
          detail: "Render state: Idle"
        }),
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

  it("uses live voice reason codes for recovery-oriented diagnostics copy", () => {
    const diagnostics = buildPersonaBuddyDiagnostics({
      selectedPersona: { id: "persona-1", name: "Ada" },
      profileState: "loaded",
      buddySummary: "Ready",
      capabilities: { hasPersona: true, hasMcp: true },
      liveSession: { connected: true, connecting: false, sessionId: "session-1" },
      liveVoice: {
        state: "idle",
        recoveryMode: "none",
        warning:
          "Server VAD unavailable for this live session. Use Send now to commit heard speech manually.",
        warningReasonCode: "voice_manual_mode_required",
        manualModeRequired: true
      },
      wake: { armed: false, detectorState: "idle" },
      visual: { packLoadStatus: "loaded", diagnostic: null }
    })

    expect(diagnostics.state).toBe("degraded")
    expect(diagnostics.rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: "Live voice",
          value: "Manual commit required",
          state: "degraded",
          detail: expect.stringMatching(/manual controls remain available/i)
        })
      ])
    )
  })

  it("uses wake reason codes without treating disabled wake as broken Persona Live", () => {
    const permissionNeeded = buildPersonaBuddyDiagnostics({
      selectedPersona: { id: "persona-1", name: "Ada" },
      profileState: "loaded",
      buddySummary: "Ready",
      capabilities: { hasPersona: true, hasMcp: true },
      liveSession: { connected: true, connecting: false, sessionId: "session-1" },
      liveVoice: { state: "idle", recoveryMode: "none" },
      wake: {
        armed: true,
        detectorState: "error",
        warning: "Microphone permission is blocked.",
        warningReasonCode: "wake_detector_permission_denied"
      },
      visual: { packLoadStatus: "loaded", diagnostic: null }
    })

    expect(permissionNeeded.state).toBe("degraded")
    expect(permissionNeeded.rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: "Wake",
          value: "Permission needed",
          state: "degraded",
          detail: expect.stringMatching(/manual controls remain available/i)
        })
      ])
    )

    const notConfigured = buildPersonaBuddyDiagnostics({
      selectedPersona: { id: "persona-1", name: "Ada" },
      profileState: "loaded",
      buddySummary: "Ready",
      capabilities: { hasPersona: true, hasMcp: true },
      liveSession: { connected: true, connecting: false, sessionId: "session-1" },
      liveVoice: { state: "idle", recoveryMode: "none" },
      wake: {
        armed: false,
        detectorState: "idle",
        warning: "Add a persona trigger phrase before arming wake listening.",
        warningReasonCode: "wake_not_configured"
      },
      visual: { packLoadStatus: "loaded", diagnostic: null }
    })

    expect(notConfigured.rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: "Wake",
          value: "Not configured",
          state: "healthy",
          detail: expect.stringMatching(/manual controls remain available/i)
        })
      ])
    )
    expect(notConfigured.state).toBe("healthy")
  })

  it("does not mark intentionally dormant Buddy summaries as degraded", () => {
    const diagnostics = buildPersonaBuddyDiagnostics({
      selectedPersona: { id: "persona-1", name: "Ada" },
      profileState: "loaded",
      buddySummary: null,
      capabilities: { hasPersona: true, hasMcp: true },
      liveSession: { connected: false, connecting: false, sessionId: null },
      liveVoice: { state: "idle", recoveryMode: "none" },
      wake: { armed: false, detectorState: "idle" },
      visual: { packLoadStatus: "idle", diagnostic: null }
    })

    expect(diagnostics.rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: "Buddy",
          value: "Dormant",
          state: "healthy"
        })
      ])
    )
    expect(diagnostics.state).toBe("healthy")
  })

  it("treats unconfirmed MCP and Persona capability readiness as degraded or unavailable", () => {
    const unknownMcp = buildPersonaBuddyDiagnostics({
      selectedPersona: { id: "persona-1", name: "Ada" },
      profileState: "loaded",
      buddySummary: "Ready",
      capabilities: { hasPersona: true },
      liveSession: { connected: false, connecting: false, sessionId: null },
      liveVoice: { state: "idle", recoveryMode: "none" },
      wake: { armed: false, detectorState: "idle" },
      visual: { packLoadStatus: "idle", diagnostic: null }
    })

    expect(unknownMcp.rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: "MCP persona_visuals",
          value: "Unknown",
          state: "degraded"
        })
      ])
    )
    expect(unknownMcp.state).toBe("degraded")

    const unknownPersonaCapability = buildPersonaBuddyDiagnostics({
      selectedPersona: { id: "persona-1", name: "Ada" },
      profileState: "loaded",
      buddySummary: "Ready",
      capabilities: { hasMcp: true },
      liveSession: { connected: false, connecting: false, sessionId: null },
      liveVoice: { state: "idle", recoveryMode: "none" },
      wake: { armed: false, detectorState: "idle" },
      visual: { packLoadStatus: "idle", diagnostic: null }
    })

    expect(unknownPersonaCapability.rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: "Server capability",
          value: "Persona unavailable",
          state: "unavailable"
        })
      ])
    )
    expect(unknownPersonaCapability.state).toBe("unavailable")
  })

  it("surfaces live session last events in the session diagnostics row", () => {
    const diagnostics = buildPersonaBuddyDiagnostics({
      selectedPersona: { id: "persona-1", name: "Ada" },
      profileState: "loaded",
      buddySummary: "Ready",
      capabilities: { hasPersona: true, hasMcp: true },
      liveSession: {
        connected: true,
        connecting: false,
        sessionId: "session-1",
        lastEvent: "ws_open"
      },
      liveVoice: { state: "idle", recoveryMode: "none" },
      wake: { armed: false, detectorState: "idle" },
      visual: { packLoadStatus: "idle", diagnostic: null }
    })

    expect(diagnostics.rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: "Live session",
          detail: "Session session-1 - Last event: ws_open"
        })
      ])
    )
  })
})
