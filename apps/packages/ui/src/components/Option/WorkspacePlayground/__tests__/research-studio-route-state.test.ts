import { describe, expect, it } from "vitest"
import {
  RESEARCH_STUDIO_LAST_MOBILE_TAB_STORAGE_KEY,
  getInitialResearchStudioTab,
  getResearchStudioTabFromSearch,
  parseResearchStudioTab,
  readResearchStudioLastMobileTab,
  writeResearchStudioLastMobileTab
} from "../research-studio-route-state"

describe("Research Studio route state", () => {
  it("accepts only canonical Research Studio tab values", () => {
    expect(parseResearchStudioTab("sources")).toBe("sources")
    expect(parseResearchStudioTab("chat")).toBe("chat")
    expect(parseResearchStudioTab("studio")).toBe("studio")

    expect(parseResearchStudioTab("Studio")).toBeNull()
    expect(parseResearchStudioTab("notes")).toBeNull()
    expect(parseResearchStudioTab("")).toBeNull()
    expect(parseResearchStudioTab(null)).toBeNull()
    expect(parseResearchStudioTab(undefined)).toBeNull()
  })

  it("reads the tab query without dropping unrelated route params", () => {
    const params = new URLSearchParams("?shared=abc&tab=studio&source=42")

    expect(getResearchStudioTabFromSearch(params)).toBe("studio")
    expect(params.get("shared")).toBe("abc")
    expect(params.get("source")).toBe("42")
  })

  it("uses the first tab param and falls back when it is invalid", () => {
    expect(getResearchStudioTabFromSearch("?tab=sources&tab=studio")).toBe(
      "sources"
    )
    expect(getResearchStudioTabFromSearch("?tab=banana&tab=studio")).toBeNull()
  })

  it("falls back to Chat for missing or invalid route state", () => {
    expect(getInitialResearchStudioTab("")).toBe("chat")
    expect(getInitialResearchStudioTab("?shared=abc")).toBe("chat")
    expect(getInitialResearchStudioTab("?tab=banana")).toBe("chat")
  })

  it("reads only valid persisted mobile tabs", () => {
    const storage = {
      getItem: (key: string) =>
        key === RESEARCH_STUDIO_LAST_MOBILE_TAB_STORAGE_KEY ? "studio" : null,
      setItem: () => undefined
    }

    expect(readResearchStudioLastMobileTab(storage)).toBe("studio")

    const invalidStorage = {
      getItem: () => "banana",
      setItem: () => undefined
    }
    expect(readResearchStudioLastMobileTab(invalidStorage)).toBeNull()
  })

  it("treats persisted tab read and write failures as no-ops", () => {
    const throwingStorage = {
      getItem: () => {
        throw new Error("blocked")
      },
      setItem: () => {
        throw new Error("blocked")
      }
    }

    expect(readResearchStudioLastMobileTab(throwingStorage)).toBeNull()
    expect(() =>
      writeResearchStudioLastMobileTab("sources", throwingStorage)
    ).not.toThrow()
  })

  it("writes valid mobile tabs to the versioned storage key", () => {
    const writes: Array<[string, string]> = []
    const storage = {
      getItem: () => null,
      setItem: (key: string, value: string) => {
        writes.push([key, value])
      }
    }

    writeResearchStudioLastMobileTab("studio", storage)

    expect(writes).toEqual([
      [RESEARCH_STUDIO_LAST_MOBILE_TAB_STORAGE_KEY, "studio"]
    ])
  })
})
