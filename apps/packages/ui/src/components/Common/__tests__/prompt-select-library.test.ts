import { describe, expect, it } from "vitest"

import type { Prompt } from "@/db/dexie/types"
import type { ServerPromptBrief } from "@/services/server-prompt-library"
import {
  matchesServerPromptSearch,
  mergePromptLibraries,
  resolveServerPromptApplication
} from "../prompt-select-library"

const local = (overrides: Partial<Prompt>): Prompt => ({
  id: "local",
  title: "Local",
  content: "Local text",
  is_system: false,
  createdAt: 1,
  ...overrides
})

const server = (id: number, name: string): ServerPromptBrief => ({
  id,
  uuid: `uuid-${id}`,
  name,
  author: null
})

describe("mergePromptLibraries", () => {
  it("collapses a copy linked by uuid even after it was renamed locally", () => {
    const merged = mergePromptLibraries(
      [local({ id: "a", title: "My tutor", serverLibraryUuid: "uuid-1" })],
      [server(1, "Socratic tutor")]
    )

    expect(merged.localEntries).toEqual([
      expect.objectContaining({ source: "both" })
    ])
    expect(merged.serverOnly).toEqual([])
  })

  it("treats a local prompt with a server prompt's unique name as its copy", () => {
    const merged = mergePromptLibraries(
      [local({ id: "a", title: "  daily STANDUP " })],
      [server(2, "Daily standup"), server(3, "Release notes")]
    )

    expect(merged.localEntries.map((entry) => entry.source)).toEqual(["both"])
    expect(merged.serverOnly.map((prompt) => prompt.id)).toEqual([3])
  })

  it("keeps unrelated prompts on both sides and sorts server prompts by name", () => {
    const merged = mergePromptLibraries(
      [local({ id: "a", title: "Local only" })],
      [server(5, "beta"), server(4, "Alpha")]
    )

    expect(merged.localEntries.map((entry) => entry.source)).toEqual(["local"])
    expect(merged.serverOnly.map((prompt) => prompt.name)).toEqual([
      "Alpha",
      "beta"
    ])
  })

  it("does not link through the Prompt Studio serverId", () => {
    const merged = mergePromptLibraries(
      [local({ id: "a", title: "Studio prompt", serverId: 1 })],
      [server(1, "Library prompt")]
    )

    expect(merged.localEntries.map((entry) => entry.source)).toEqual(["local"])
    expect(merged.serverOnly).toHaveLength(1)
  })
})

describe("matchesServerPromptSearch", () => {
  it("matches name or author, ignoring case", () => {
    const prompt = { ...server(1, "Socratic tutor"), author: "Robert" }
    expect(matchesServerPromptSearch(prompt, "SOCRATIC")).toBe(true)
    expect(matchesServerPromptSearch(prompt, "rob")).toBe(true)
    expect(matchesServerPromptSearch(prompt, "release")).toBe(false)
  })
})

describe("resolveServerPromptApplication", () => {
  it("applies system text as a system prompt, like a pulled local copy", () => {
    expect(
      resolveServerPromptApplication({
        system_prompt: "Be terse.",
        user_prompt: "Summarise"
      })
    ).toEqual({ kind: "system", content: "Be terse." })
  })

  it("applies user text as a quick prompt when there is no system text", () => {
    expect(
      resolveServerPromptApplication({ system_prompt: " ", user_prompt: "Summarise" })
    ).toEqual({ kind: "quick", content: "Summarise" })
  })

  it("reports a prompt with no text", () => {
    expect(
      resolveServerPromptApplication({ system_prompt: null, user_prompt: null })
    ).toEqual({ kind: "empty" })
  })
})
