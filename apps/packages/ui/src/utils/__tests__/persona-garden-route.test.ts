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
    expect(buildPersonaGardenRoute({ tab: "profiles" })).toBe(
      "/persona?tab=profiles"
    )
  })

  it("parses persona garden bootstrap params from search", () => {
    expect(
      readPersonaGardenSearch("?persona_id=garden-helper&tab=profiles")
    ).toEqual({
      personaId: "garden-helper",
      sessionId: null,
      tab: "profiles"
    })
  })

  it("ignores invalid persona garden tab values", () => {
    expect(
      readPersonaGardenSearch("?persona_id=garden-helper&tab=unknown")
    ).toEqual({
      personaId: "garden-helper",
      sessionId: null,
      tab: null
    })
  })

  it("accepts the voice tab in persona garden routes", () => {
    expect(
      readPersonaGardenSearch("?persona_id=garden-helper&tab=voice")
    ).toEqual({
      personaId: "garden-helper",
      sessionId: null,
      tab: "voice"
    })
  })

  it("accepts the commands and connections tabs in persona garden routes", () => {
    expect(
      readPersonaGardenSearch("?persona_id=garden-helper&tab=commands")
    ).toEqual({
      personaId: "garden-helper",
      sessionId: null,
      tab: "commands"
    })
    expect(
      readPersonaGardenSearch("?persona_id=garden-helper&tab=connections")
    ).toEqual({
      personaId: "garden-helper",
      sessionId: null,
      tab: "connections"
    })
  })

  it("accepts the visuals tab in persona garden routes", () => {
    expect(buildPersonaGardenRoute({ tab: "visuals" })).toBe(
      "/persona?tab=visuals"
    )
    expect(
      readPersonaGardenSearch("?persona_id=garden-helper&tab=visuals")
    ).toEqual({
      personaId: "garden-helper",
      sessionId: null,
      tab: "visuals"
    })
  })
})

it("round trips exact live session identity without plan contents", () => {
  const route = buildPersonaGardenRoute({
    personaId: "p",
    tab: "live",
    sessionId: "sess /1"
  })
  expect(route).toBe("/persona?persona_id=p&tab=live&session_id=sess+%2F1")
  expect(readPersonaGardenSearch(route.split("?")[1])).toEqual({
    personaId: "p",
    tab: "live",
    sessionId: "sess /1"
  })
  expect(
    readPersonaGardenSearch("?session_id=s&tab=profiles&plan=secret").sessionId
  ).toBeNull()
  expect(buildPersonaGardenRoute({ tab: "profiles", sessionId: "s" })).toBe(
    "/persona?tab=profiles"
  )
})
