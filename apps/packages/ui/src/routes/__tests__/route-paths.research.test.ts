import { describe, expect, it } from "vitest"
import {
  CHAT_PATH,
  RESEARCH_PATH,
  buildCharacterChatPath,
  buildChatThreadPath,
  buildResearchLaunchPath
} from "../route-paths"
import { SETTINGS_SERVER_CHAT_ID_PARAM } from "../../utils/settings-return"

describe("route-paths deep research launch", () => {
  it("exports the canonical research path", () => {
    expect(RESEARCH_PATH).toBe("/research")
  })

  it("builds a launch path with encoded query and launch options", () => {
    const href = buildResearchLaunchPath({
      query: "Investigate local evidence & timeline",
      sourcePolicy: "balanced",
      autonomyMode: "checkpointed",
      autorun: true,
      from: "chat",
      chatId: "chat_123"
    })
    const parsed = new URL(href, "https://example.local")

    expect(parsed.pathname).toBe(RESEARCH_PATH)
    expect(parsed.searchParams.get("query")).toBe(
      "Investigate local evidence & timeline"
    )
    expect(parsed.searchParams.get("source_policy")).toBe("balanced")
    expect(parsed.searchParams.get("autonomy_mode")).toBe("checkpointed")
    expect(parsed.searchParams.get("autorun")).toBe("1")
    expect(parsed.searchParams.get("from")).toBe("chat")
    expect(parsed.searchParams.get("chat_id")).toBe("chat_123")
  })

  it("omits empty launch fields", () => {
    const href = buildResearchLaunchPath({
      query: "   ",
      sourcePolicy: "",
      autonomyMode: "",
      autorun: false,
      from: ""
    })
    const parsed = new URL(href, "https://example.local")

    expect(parsed.pathname).toBe(RESEARCH_PATH)
    expect(parsed.searchParams.get("query")).toBeNull()
    expect(parsed.searchParams.get("source_policy")).toBeNull()
    expect(parsed.searchParams.get("autonomy_mode")).toBeNull()
    expect(parsed.searchParams.get("autorun")).toBeNull()
    expect(parsed.searchParams.get("from")).toBeNull()
  })

  it("builds an exact chat-thread path for server-backed return handoff", () => {
    const href = buildChatThreadPath({
      serverChatId: "chat_123",
      researchReturnRunId: "rs_1"
    })
    const parsed = new URL(href, "https://example.local")

    expect(parsed.pathname).toBe(CHAT_PATH)
    expect(parsed.searchParams.get(SETTINGS_SERVER_CHAT_ID_PARAM)).toBe("chat_123")
    expect(parsed.searchParams.get("researchReturnRunId")).toBe("rs_1")
  })

  it("builds a first-class character chat intent path", () => {
    const href = buildCharacterChatPath({ characterId: "char 123" })
    const parsed = new URL(href, "https://example.local")

    expect(parsed.pathname).toBe(CHAT_PATH)
    expect(parsed.searchParams.get("mode")).toBe("character")
    expect(parsed.searchParams.get("characterId")).toBe("char 123")
  })
})
