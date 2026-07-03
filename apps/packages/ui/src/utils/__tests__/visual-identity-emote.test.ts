import { describe, expect, it } from "vitest"
import { parseVisualIdentityEmoteCommand } from "../visual-identity-emote"

describe("visual identity emote command parsing", () => {
  it("maps anger slash command to angry expression", () => {
    expect(parseVisualIdentityEmoteCommand("/emote anger")).toEqual({
      expressionKey: "angry",
      rawExpression: "anger"
    })
  })

  it("keeps regular messages untouched", () => {
    expect(parseVisualIdentityEmoteCommand("please /emote happy")).toBeNull()
  })

  it("normalizes custom labels in slash commands", () => {
    expect(parseVisualIdentityEmoteCommand("/emote bashful smile")).toEqual({
      expressionKey: "custom:bashful_smile",
      rawExpression: "bashful smile"
    })
  })

  it("ignores empty emote commands", () => {
    expect(parseVisualIdentityEmoteCommand("/emote")).toBeNull()
  })
})
