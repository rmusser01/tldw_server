import { describe, expect, it } from "vitest"

import { searchSettings } from "@/data/settings-index"

const translate = (_key: string, defaultValue?: string) => defaultValue ?? ""

describe("settings index RAG discoverability", () => {
  it("finds Persona Garden without replacing Characters", () => {
    const personaResults = searchSettings("persona garden", translate)
    const characterResults = searchSettings("characters", translate)

    expect(
      personaResults.some(
        (setting) => setting.id === "setting-persona-garden" && setting.route === "/persona"
      )
    ).toBe(true)
    expect(
      characterResults.some(
        (setting) => setting.id === "setting-characters" && setting.route === "/settings/characters"
      )
    ).toBe(true)
  })

  it("finds the unified RAG defaults page by knowledge QA query", () => {
    const results = searchSettings("knowledge qa", translate)

    expect(
      results.some(
        (setting) =>
          setting.id === "setting-rag-default-profile" &&
          setting.route === "/settings/rag"
      )
    ).toBe(true)
  })

  it("finds chat context window controls", () => {
    const results = searchSettings("context window", translate)

    expect(
      results.some((setting) => setting.id === "setting-rag-chat-max-context")
    ).toBe(true)
  })

  it("sends reusable Prompt Library searches to the workspace", () => {
    const results = searchSettings("prompt library", translate)
    const promptLibrary = results.find((setting) => setting.id === "setting-prompts")

    expect(promptLibrary?.route).toBe("/prompts")
    expect(promptLibrary?.defaultLabel).toBe("Prompt Library")
  })
})
