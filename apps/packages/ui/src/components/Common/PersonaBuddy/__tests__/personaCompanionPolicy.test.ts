import { describe, expect, it } from "vitest"

import { asPersonaVisualCustomStateId } from "@/types/persona-visuals"
import {
  resolveEffectiveAmbientMode,
  resolveWinningPersonaVisualIntent
} from "../personaCompanionPolicy"

describe("resolveEffectiveAmbientMode", () => {
  it("uses persona, then global, then Expressive", () => {
    expect(
      resolveEffectiveAmbientMode({
        persona: "roaming",
        global: "off",
        readFailed: false,
        surface: "web"
      })
    ).toBe("roaming")
    expect(
      resolveEffectiveAmbientMode({
        persona: null,
        global: "off",
        readFailed: false,
        surface: "web"
      })
    ).toBe("off")
    expect(
      resolveEffectiveAmbientMode({
        persona: null,
        global: null,
        readFailed: false,
        surface: "web"
      })
    ).toBe("expressive")
  })

  it("fails closed and coerces sidepanel roaming", () => {
    expect(
      resolveEffectiveAmbientMode({
        persona: "roaming",
        global: "expressive",
        readFailed: true,
        surface: "web"
      })
    ).toBe("off")
    expect(
      resolveEffectiveAmbientMode({
        persona: "roaming",
        global: null,
        readFailed: false,
        surface: "sidepanel"
      })
    ).toBe("expressive")
  })
})

describe("resolveWinningPersonaVisualIntent", () => {
  it("uses error, approval, offline, active, interaction, ambient, then idle precedence", () => {
    const custom = asPersonaVisualCustomStateId("reaction.click")
    const input = {
      error: "error" as const,
      approval: "approval_needed" as const,
      offline: "offline" as const,
      wake: "wake_armed" as const,
      listening: "listening" as const,
      thinking: "thinking" as const,
      speaking: "speaking" as const,
      tool: "tool_running" as const,
      interaction: custom,
      ambient: asPersonaVisualCustomStateId("ambient.look"),
      idle: "idle" as const
    }

    expect(resolveWinningPersonaVisualIntent(input)).toBe("error")
    expect(resolveWinningPersonaVisualIntent({ ...input, error: null })).toBe(
      "approval_needed"
    )
    expect(
      resolveWinningPersonaVisualIntent({ ...input, error: null, approval: null })
    ).toBe("offline")
    expect(
      resolveWinningPersonaVisualIntent({
        ...input,
        error: null,
        approval: null,
        offline: null
      })
    ).toBe("wake_armed")
    expect(
      resolveWinningPersonaVisualIntent({
        ...input,
        error: null,
        approval: null,
        offline: null,
        wake: null
      })
    ).toBe("listening")
    expect(
      resolveWinningPersonaVisualIntent({
        ...input,
        error: null,
        approval: null,
        offline: null,
        wake: null,
        listening: null
      })
    ).toBe("thinking")
    expect(
      resolveWinningPersonaVisualIntent({
        ...input,
        error: null,
        approval: null,
        offline: null,
        wake: null,
        listening: null,
        thinking: null,
      })
    ).toBe("speaking")
    expect(
      resolveWinningPersonaVisualIntent({
        ...input,
        error: null,
        approval: null,
        offline: null,
        wake: null,
        listening: null,
        thinking: null,
        speaking: null
      })
    ).toBe("tool_running")
    expect(
      resolveWinningPersonaVisualIntent({
        ...input,
        error: null,
        approval: null,
        offline: null,
        wake: null,
        listening: null,
        thinking: null,
        speaking: null,
        tool: null
      })
    ).toBe("reaction.click")
    expect(
      resolveWinningPersonaVisualIntent({
        ...input,
        error: null,
        approval: null,
        offline: null,
        wake: null,
        listening: null,
        thinking: null,
        speaking: null,
        tool: null,
        interaction: null
      })
    ).toBe("ambient.look")
    expect(resolveWinningPersonaVisualIntent({})).toBe("idle")
  })
})
