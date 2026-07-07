import { describe, expect, it } from "vitest"
import {
  buildCharactersHash,
  buildCharactersRoute,
  resolveCharactersDestinationMode
} from "../characters-route"

describe("characters-route", () => {
  it("builds characters route with source parameter", () => {
    expect(buildCharactersRoute({ from: "header-select" })).toBe(
      "/characters?from=header-select"
    )
  })

  it("builds characters route with create=true flag", () => {
    expect(
      buildCharactersRoute({
        from: "header-select",
        create: true
      })
    ).toBe("/characters?from=header-select&create=true")
  })

  it("builds characters route with expression focus", () => {
    expect(
      buildCharactersRoute({
        from: "sidepanel-character-controls",
        focus: "expressions",
        characterId: 42
      })
    ).toBe(
      "/characters?from=sidepanel-character-controls&focus=expressions&characterId=42"
    )
  })

  it("builds hash route for extension options navigation", () => {
    expect(
      buildCharactersHash({
        from: "header-select",
        create: true
      })
    ).toBe("#/characters?from=header-select&create=true")
  })

  it("resolves in-place mode for options page path", () => {
    expect(
      resolveCharactersDestinationMode({
        pathname: "/options.html",
        extensionRuntime: false
      })
    ).toBe("options-in-place")
  })

  it("resolves extension tab mode outside options page", () => {
    expect(
      resolveCharactersDestinationMode({
        pathname: "/chat",
        extensionRuntime: true
      })
    ).toBe("options-tab")
  })

  it("resolves direct web route mode when not in extension runtime", () => {
    expect(
      resolveCharactersDestinationMode({
        pathname: "/chat",
        extensionRuntime: false
      })
    ).toBe("web-route")
  })
})
