import { describe, expect, it } from "vitest"
import {
  RESEARCH_WORKSPACE_LAST_MOBILE_TAB_STORAGE_KEY,
  getInitialResearchWorkspaceTab,
  getResearchWorkspaceSearchFromLocation,
  getResearchWorkspaceTabFromSearch,
  parseResearchWorkspaceTab,
  readResearchWorkspaceLastMobileTab,
  writeResearchWorkspaceLastMobileTab
} from "../research-workspace-route-state"

describe("Research Workspace route state", () => {
  it("accepts only canonical Research Workspace tab values", () => {
    expect(parseResearchWorkspaceTab("sources")).toBe("sources")
    expect(parseResearchWorkspaceTab("chat")).toBe("chat")
    expect(parseResearchWorkspaceTab("studio")).toBe("studio")

    expect(parseResearchWorkspaceTab("Studio")).toBeNull()
    expect(parseResearchWorkspaceTab("notes")).toBeNull()
    expect(parseResearchWorkspaceTab("")).toBeNull()
    expect(parseResearchWorkspaceTab(null)).toBeNull()
    expect(parseResearchWorkspaceTab(undefined)).toBeNull()
  })

  it("reads the tab query without dropping unrelated route params", () => {
    const params = new URLSearchParams("?shared=abc&tab=studio&source=42")

    expect(getResearchWorkspaceTabFromSearch(params)).toBe("studio")
    expect(params.get("shared")).toBe("abc")
    expect(params.get("source")).toBe("42")
  })

  it("uses the first tab param and falls back when it is invalid", () => {
    expect(getResearchWorkspaceTabFromSearch("?tab=sources&tab=studio")).toBe(
      "sources"
    )
    expect(getResearchWorkspaceTabFromSearch("?tab=banana&tab=studio")).toBeNull()
  })

  it("reads query params from normal and hash-router locations", () => {
    expect(
      getResearchWorkspaceSearchFromLocation({
        search: "?tab=studio",
        hash: "#/research-workspace?tab=sources"
      })
    ).toBe("?tab=studio")
    expect(
      getResearchWorkspaceSearchFromLocation({
        search: "",
        hash: "#/research-workspace?tab=studio&shared=123"
      })
    ).toBe("?tab=studio&shared=123")
    expect(
      getResearchWorkspaceSearchFromLocation({
        search: "",
        hash: "#/research-workspace"
      })
    ).toBe("")
  })

  it("falls back to Chat for missing or invalid route state", () => {
    expect(getInitialResearchWorkspaceTab("")).toBe("chat")
    expect(getInitialResearchWorkspaceTab("?shared=abc")).toBe("chat")
    expect(getInitialResearchWorkspaceTab("?tab=banana")).toBe("chat")
  })

  it("reads only valid persisted mobile tabs", () => {
    const storage = {
      getItem: (key: string) =>
        key === RESEARCH_WORKSPACE_LAST_MOBILE_TAB_STORAGE_KEY ? "studio" : null,
      setItem: () => undefined
    }

    expect(readResearchWorkspaceLastMobileTab(storage)).toBe("studio")

    const invalidStorage = {
      getItem: () => "banana",
      setItem: () => undefined
    }
    expect(readResearchWorkspaceLastMobileTab(invalidStorage)).toBeNull()
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

    expect(readResearchWorkspaceLastMobileTab(throwingStorage)).toBeNull()
    expect(() =>
      writeResearchWorkspaceLastMobileTab("sources", throwingStorage)
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

    writeResearchWorkspaceLastMobileTab("studio", storage)

    expect(writes).toEqual([
      [RESEARCH_WORKSPACE_LAST_MOBILE_TAB_STORAGE_KEY, "studio"]
    ])
  })
})
