import { describe, expect, it } from "vitest"

import {
  buildPersonaGardenRoute,
  readPersonaGardenSearch
} from "../persona-garden-route"

describe("persona-garden-route", () => {
  it("builds a persona garden route with persona_id and tab", () => {
    expect(
      buildPersonaGardenRoute({
        personaId: "garden-helper",
        tab: "profiles"
      })
    ).toBe("/persona?persona_id=garden-helper&tab=profiles")
  })

  it("builds a persona garden route with only a tab", () => {
    expect(buildPersonaGardenRoute({ tab: "profiles" })).toBe("/persona?tab=profiles")
  })

  it("parses persona garden bootstrap params from search", () => {
    expect(
      readPersonaGardenSearch("?persona_id=garden-helper&tab=profiles")
    ).toEqual({
      personaId: "garden-helper",
      tab: "profiles"
    })
  })

  it("ignores invalid persona garden tab values", () => {
    expect(readPersonaGardenSearch("?persona_id=garden-helper&tab=unknown")).toEqual({
      personaId: "garden-helper",
      tab: null
    })
  })

  it("accepts the voice tab in persona garden routes", () => {
    expect(
      readPersonaGardenSearch("?persona_id=garden-helper&tab=voice")
    ).toEqual({
      personaId: "garden-helper",
      tab: "voice"
    })
  })

  it("accepts the commands and connections tabs in persona garden routes", () => {
    expect(
      readPersonaGardenSearch("?persona_id=garden-helper&tab=commands")
    ).toEqual({
      personaId: "garden-helper",
      tab: "commands"
    })
    expect(
      readPersonaGardenSearch("?persona_id=garden-helper&tab=connections")
    ).toEqual({
      personaId: "garden-helper",
      tab: "connections"
    })
  })

  it("accepts the visuals tab in persona garden routes", () => {
    expect(buildPersonaGardenRoute({ tab: "visuals" })).toBe("/persona?tab=visuals")
    expect(
      readPersonaGardenSearch("?persona_id=garden-helper&tab=visuals")
    ).toEqual({
      personaId: "garden-helper",
      tab: "visuals"
    })
  })
})
