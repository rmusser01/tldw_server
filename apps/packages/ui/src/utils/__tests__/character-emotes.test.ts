import { describe, expect, it } from "vitest"
import fixtures from "../__fixtures__/character-emote-directives.json"
import {
  EMOTE_EVENT_LIMIT,
  createCharacterEmoteStreamParser,
  isValidCharacterEmoteEvent,
  normalizeCharacterEmoteState,
  parseCharacterEmoteDirectives
} from "../character-emotes"

describe("character emote directives", () => {
  it.each(fixtures)("$name", (fixture) => {
    expect(parseCharacterEmoteDirectives(fixture.input)).toEqual({
      cleanText: fixture.clean_text,
      events: fixture.events
    })
  })

  it.each(fixtures)("streams fixture parity: $name", (fixture) => {
    const parser = createCharacterEmoteStreamParser()
    const sizes = [1, 2, 5, 3]
    const events = []
    let cleanText = ""
    let offset = 0

    for (let index = 0; offset < fixture.input.length; index += 1) {
      const size = sizes[index % sizes.length]
      const pushed = parser.push(fixture.input.slice(offset, offset + size))
      cleanText += pushed.visibleText
      events.push(...pushed.events)
      offset += size
    }

    const flushed = parser.flush()
    cleanText += flushed.visibleText
    events.push(...flushed.events)

    expect({ cleanText, events }).toEqual({
      cleanText: fixture.clean_text,
      events: fixture.events
    })
    expect(parser.flush()).toEqual({ visibleText: "", events: [] })
  })

  it("normalizes emote states exactly", () => {
    expect(normalizeCharacterEmoteState(" Thinking Hard ")).toBe("thinking-hard")
    expect(normalizeCharacterEmoteState("../../bad")).toBeNull()
    expect(normalizeCharacterEmoteState("a".repeat(40))).toBe("a".repeat(40))
    expect(normalizeCharacterEmoteState("a".repeat(41))).toBeNull()
  })

  it("caps accepted events while stripping later directives", () => {
    const directives = Array.from(
      { length: EMOTE_EVENT_LIMIT + 2 },
      (_, index) => `Emote: state-${index}`
    )
    const parsed = parseCharacterEmoteDirectives(`${directives.join("\n")}\nDone.`)

    expect(parsed.cleanText).toBe("Done.")
    expect(parsed.events).toEqual(
      Array.from({ length: EMOTE_EVENT_LIMIT }, (_, index) => ({
        state: `state-${index}`,
        at_char: 0
      }))
    )
  })

  it("does not leak a directive split across streaming chunks", () => {
    const parser = createCharacterEmoteStreamParser()

    expect(parser.push("Em")).toEqual({ visibleText: "", events: [] })
    expect(parser.push("ote: smug\nHello")).toEqual({
      visibleText: "Hello",
      events: [{ state: "smug", at_char: 0 }]
    })
    expect(parser.flush()).toEqual({ visibleText: "", events: [] })
  })

  it("streams long non-directive text before newline", () => {
    const parser = createCharacterEmoteStreamParser()
    const text = "This ordinary prose has no newline yet."

    expect(parser.push(text)).toEqual({ visibleText: text, events: [] })
  })

  it("validates compact emote event metadata", () => {
    expect(isValidCharacterEmoteEvent({ state: "smug", at_char: 0 })).toBe(true)
    expect(isValidCharacterEmoteEvent({ state: "../../bad", at_char: 0 })).toBe(false)
    expect(isValidCharacterEmoteEvent({ state: "smug", at_char: -1 })).toBe(false)
  })
})
