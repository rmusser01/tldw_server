import { describe, expect, it } from "vitest"

import { asPersonaVisualCustomStateId } from "@/types/persona-visuals"
import { resolvePersonaVisualState } from "../personaVisualState"

describe("resolvePersonaVisualState", () => {
  it("prefers error over live voice state", () => {
    expect(
      resolvePersonaVisualState({
        liveVoiceState: "speaking",
        hasError: true
      })
    ).toBe("error")
  })

  it("prefers approval-needed over tool and voice state", () => {
    expect(
      resolvePersonaVisualState({
        approvalNeeded: true,
        activeToolStatus: "Running notes.search",
        liveVoiceState: "thinking"
      })
    ).toBe("approval_needed")
  })

  it("uses an unexpired runtime override before authored triggers", () => {
    expect(
      resolvePersonaVisualState({
        liveVoiceState: "speaking",
        activeToolStatus: "Running notes.search",
        runtimeOverride: {
          state: "approval_needed",
          reason: "mcp-trigger",
          expiresAt: 2000
        },
        now: 1000
      })
    ).toBe("approval_needed")
  })

  it("ignores expired runtime overrides", () => {
    expect(
      resolvePersonaVisualState({
        liveVoiceState: "speaking",
        runtimeOverride: {
          state: "error",
          reason: "old-trigger",
          expiresAt: 1000
        },
        now: 2000
      })
    ).toBe("speaking")
  })

  it("uses the highest-priority matching authored trigger", () => {
    expect(
      resolvePersonaVisualState({
        liveVoiceState: "thinking",
        activeToolStatus: "Running notes.search",
        authoredTriggers: [
          {
            id: "low",
            source: "live_state",
            match: "thinking",
            state: "thinking",
            duration_ms: 500,
            priority: 10
          },
          {
            id: "high",
            source: "tool_category",
            match: "notes",
            state: "approval_needed",
            duration_ms: 500,
            priority: 90
          }
        ]
      })
    ).toBe("approval_needed")
  })

  it("uses exact tool_name triggers from structured tool context", () => {
    const customState = asPersonaVisualCustomStateId("tool.notes_search")
    expect(
      resolvePersonaVisualState({
        liveVoiceState: "thinking",
        activeToolName: "notes.search",
        activeToolStatus: "Searching notes",
        authoredTriggers: [
          {
            id: "notes-search",
            source: "tool_name",
            match: "notes.search",
            state: customState,
            duration_ms: 500,
            priority: 90
          }
        ]
      })
    ).toBe("tool.notes_search")
  })

  it("does not infer exact tool_name triggers from status display text", () => {
    const customState = asPersonaVisualCustomStateId("tool.notes_search")
    expect(
      resolvePersonaVisualState({
        liveVoiceState: "thinking",
        activeToolStatus: "Running notes.search",
        authoredTriggers: [
          {
            id: "notes-search",
            source: "tool_name",
            match: "notes.search",
            state: customState,
            duration_ms: 500,
            priority: 90
          }
        ]
      })
    ).toBe("tool_running")
  })

  it("maps active tool status to tool_running", () => {
    expect(
      resolvePersonaVisualState({
        liveVoiceState: "thinking",
        activeToolStatus: "Running notes.search"
      })
    ).toBe("tool_running")
  })

  it("maps wake armed before idle", () => {
    expect(
      resolvePersonaVisualState({
        liveVoiceState: "idle",
        wakeArmed: true
      })
    ).toBe("wake_armed")
  })

  it("maps missing online voice state to offline when offline", () => {
    expect(
      resolvePersonaVisualState({
        isOffline: true
      })
    ).toBe("offline")
  })
})
