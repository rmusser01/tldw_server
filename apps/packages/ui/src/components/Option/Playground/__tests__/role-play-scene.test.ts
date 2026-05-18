import { describe, expect, it } from "vitest"

import { createDefaultActorSettings } from "@/types/actor"
import {
  clearRolePlayScene,
  resetRolePlayScene,
  summarizeRolePlayScene
} from "../role-play-scene"

const enabledScene = () => {
  const settings = createDefaultActorSettings()
  return {
    ...settings,
    isEnabled: true,
    notes: "The market is crowded and tense.",
    aspects: settings.aspects.map((aspect) =>
      aspect.id === "world_location"
        ? { ...aspect, value: "Harbor market" }
        : aspect.id === "char_state"
          ? { ...aspect, value: "watchful" }
          : aspect
    )
  }
}

describe("role-play scene adapter", () => {
  it("summarizes default actor settings as inactive scene", () => {
    const preview = summarizeRolePlayScene(createDefaultActorSettings())

    expect(preview).toEqual({
      active: false,
      summary: "No scene",
      prompt: "",
      tokenCount: 0
    })
  })

  it("summarizes enabled notes and aspects as an active scene", () => {
    const preview = summarizeRolePlayScene(enabledScene())

    expect(preview.active).toBe(true)
    expect(preview.summary).toContain("2 details")
    expect(preview.summary).toContain("notes")
    expect(preview.prompt).toContain("The location is Harbor market.")
    expect(preview.prompt).toContain("{{char}}'s character emotional state is watchful.")
    expect(preview.prompt).toContain("Scene notes: The market is crowded and tense.")
    expect(preview.tokenCount).toBeGreaterThan(0)
  })

  it("excludes GM-only notes from the prompt preview", () => {
    const settings = {
      ...enabledScene(),
      notes: "Do not reveal the ambush.",
      notesGmOnly: true
    }

    const preview = summarizeRolePlayScene(settings)

    expect(preview.active).toBe(true)
    expect(preview.summary).toContain("GM-only notes")
    expect(preview.prompt).not.toContain("Do not reveal the ambush.")
  })

  it("clears scene by disabling actor and emptying notes and aspects", () => {
    const cleared = clearRolePlayScene(enabledScene())

    expect(cleared.isEnabled).toBe(false)
    expect(cleared.notes).toBe("")
    expect(cleared.notesGmOnly).toBe(false)
    expect(cleared.aspects.every((aspect) => aspect.value === "")).toBe(true)
  })

  it("resets scene to createDefaultActorSettings", () => {
    expect(resetRolePlayScene()).toEqual(createDefaultActorSettings())
  })
})
