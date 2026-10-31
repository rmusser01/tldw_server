import { describe, expect, it } from "vitest"
import {
  buildGreetingOptionsFromEntries,
  buildGreetingsChecksumFromOptions,
  collectGreetingEntries,
  collectGreetings,
  normalizeGreetingValue,
  parseGreetingSelectionIndex,
  resolveGreetingSelection,
  resolveCharacterMessageType
} from "../character-greetings"

describe("character greetings utils", () => {
  it("parses JSON-string greeting lists", () => {
    expect(normalizeGreetingValue('["Hi there","Welcome back"]')).toEqual([
      "Hi there",
      "Welcome back"
    ])
  })

  it("parses double-encoded JSON-string greeting lists", () => {
    expect(
      normalizeGreetingValue('"[\\"One\\", \\"Two\\", \\"Three\\"]"')
    ).toEqual(["One", "Two", "Three"])
  })

  it("keeps multiline single greetings intact", () => {
    expect(normalizeGreetingValue("Hello there.\nHow are you today?")).toEqual([
      "Hello there.\nHow are you today?"
    ])
  })

  it("collects primary and alternate greetings from mixed payloads", () => {
    const character = {
      first_message: "Hello!",
      alternate_greetings: '["Hey there!","Welcome!"]'
    }
    expect(collectGreetings(character)).toEqual([
      "Hello!",
      "Hey there!",
      "Welcome!"
    ])

    const entries = collectGreetingEntries(character)
    expect(entries.map((entry) => entry.text)).toEqual([
      "Hello!",
      "Hey there!",
      "Welcome!"
    ])
  })

  it("parses greeting selection indices from id formats", () => {
    expect(parseGreetingSelectionIndex("greeting:2:abc123")).toBe(2)
    expect(parseGreetingSelectionIndex("greeting:4:selected")).toBe(4)
    expect(parseGreetingSelectionIndex("greeting:-1:selected")).toBeNull()
    expect(parseGreetingSelectionIndex("invalid")).toBeNull()
  })

  it("resolves greeting by stored selection id", () => {
    const character = {
      greeting: "Primary",
      alternateGreetings: ["Secondary"]
    }
    const options = buildGreetingOptionsFromEntries(
      collectGreetingEntries(character)
    )
    const checksum = buildGreetingsChecksumFromOptions(options)
    const selectedOption = options[1]
    if (!selectedOption) {
      throw new Error("Expected alternate greeting option")
    }

    const result = resolveGreetingSelection({
      options,
      greetingSelectionId: selectedOption.id,
      greetingsChecksum: checksum,
      useCharacterDefault: false,
      fallback: "first"
    })

    expect(result.option?.text).toBe("Secondary")
    expect(result.isStale).toBe(false)
  })

  it("resolves greeting by legacy index id when hash differs", () => {
    const options = buildGreetingOptionsFromEntries(
      collectGreetingEntries({
        greeting: "Primary",
        alternate_greetings: ["Secondary"]
      })
    )
    const checksum = buildGreetingsChecksumFromOptions(options)

    const result = resolveGreetingSelection({
      options,
      greetingSelectionId: "greeting:1:selected",
      greetingsChecksum: checksum,
      useCharacterDefault: false,
      fallback: "first"
    })

    expect(result.option?.text).toBe("Secondary")
    expect(result.isStale).toBe(false)
  })

  it("falls back to first greeting when stored checksum is stale", () => {
    const options = buildGreetingOptionsFromEntries(
      collectGreetingEntries({
        greeting: "Primary",
        alternate_greetings: ["Secondary"]
      })
    )

    const result = resolveGreetingSelection({
      options,
      greetingSelectionId: options[1]?.id ?? null,
      greetingsChecksum: "stale-checksum",
      useCharacterDefault: false,
      fallback: "first"
    })

    expect(result.option?.text).toBe("Primary")
    expect(result.isStale).toBe(true)
  })
})


describe("native creation seed provenance is independent of display positions", () => {
  it.each([false, true])("preserves exact raw alternate indices through filtering/deduplication (JSON=%s)", json => {
    const alternates = ["", 7, "  Alternate {{char}}  ", "Alternate {{char}}", "Last"]
    const options = buildGreetingOptionsFromEntries(collectGreetingEntries({ greeting: "Compatibility alias", first_message: "  Default {{user}}  ", alternate_greetings: json ? JSON.stringify(alternates) : alternates }))
    expect(options.map(option => option.text)).toEqual(["Compatibility alias", "Default {{user}}", "Alternate {{char}}", "Last"])
    expect(options[1]).toMatchObject({ index: 1, seed: { greetingStrategy: "default", rawText: "  Default {{user}}  " } })
    expect(options[2]).toMatchObject({ index: 2, seed: { greetingStrategy: "alternate_index", alternateIndex: 2, rawText: "  Alternate {{char}}  " } })
    expect(options[3]).toMatchObject({ index: 3, seed: { greetingStrategy: "alternate_index", alternateIndex: 4, rawText: "Last" } })
    expect(options[0]).not.toHaveProperty("seed")
  })
  it("deduplicated compatibility text retains a matching canonical default before alternates", () => {
    const [option] = buildGreetingOptionsFromEntries(collectGreetingEntries({ greeting: "Same", first_message: "  Same  ", alternate_greetings: ["Same"] }))
    expect(option).toMatchObject({ sourceKey: "greeting", index: 0, seed: { greetingStrategy: "default", rawText: "  Same  " } })
    expect(option.seed).not.toHaveProperty("alternateIndex")
  })
  it.each([
    { firstMessage: "Legacy alias", alternateGreetings: ["Legacy alternate"] },
    { first_message: ["List default"], alternate_greetings: JSON.stringify(JSON.stringify(["Nested alternate"])) },
    { first_message: '["Split default"]', alternate_greetings: "Plain text alternate" }
  ])("does not manufacture a server seed for unsupported fields or encodings: %j", character => {
    const options = buildGreetingOptionsFromEntries(collectGreetingEntries(character))
    expect(options.length).toBeGreaterThan(0)
    expect(options.every(option => !Object.hasOwn(option, "seed"))).toBe(true)
  })
})

it.each(["id", "serverMessageId"] as const)("projects an owned native root greeting using %s and preserves explicit message types", idField => {
  const root = { isBot: true, role: "assistant", parentMessageId: null,
    id: idField === "id" ? "saved-root" : undefined,
    serverMessageId: idField === "serverMessageId" ? "saved-root" : undefined }
  expect(resolveCharacterMessageType(root, 0, true)).toBe("character:greeting")
  for (const [message, index, owned] of [[root, 1, true], [root, 0, false], [{ ...root, id: undefined, serverMessageId: undefined }, 0, true], [{ ...root, id: "  ", serverMessageId: "" }, 0, true], [{ ...root, parentMessageId: "user" }, 0, true], [{ ...root, isBot: false, role: "user" }, 0, true]] as const)
    expect(resolveCharacterMessageType(message, index, owned)).toBeUndefined()
  expect(resolveCharacterMessageType({ ...root, messageType: "compare:reply" }, 0, true)).toBe("compare:reply")
})
