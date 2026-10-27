import { QueryClient } from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { useNotesKeywords, type UseNotesKeywordsDeps } from "../hooks/useNotesKeywords"

// NL-08: a filter token that is not a real tag never shows up in Browse tags
// or its "Recently used" list.

vi.mock("@/services/note-keywords", () => ({
  getAllNoteKeywordStats: vi.fn(async () => [
    { keyword: "thesis", noteCount: 1 },
    { keyword: "reading", noteCount: 2 }
  ]),
  searchNoteKeywords: vi.fn(async () => [])
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn(async () => ({}))
}))

const renderKeywords = (notebookKeywordTokens: string[] = []) => {
  const deps: UseNotesKeywordsDeps = {
    isOnline: true,
    listMode: "active",
    message: {
      success: vi.fn(),
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn()
    } as unknown as UseNotesKeywordsDeps["message"],
    confirmDanger: vi.fn(async () => true),
    restoreFocusAfterOverlayClose: vi.fn(),
    notebookKeywordTokens,
    queryClient: new QueryClient(),
    t: (key, options) => (options?.defaultValue as string | undefined) ?? key,
    setPage: vi.fn(),
    setIsDirty: vi.fn(),
    setSaveIndicator: vi.fn(),
    setMonitoringNotice: vi.fn(),
    markGeneratedEdit: vi.fn()
  }
  return renderHook(() => useNotesKeywords(deps))
}

describe("useNotesKeywords NL-08 filter tokens", () => {
  it("keeps a filter token that is not a tag out of Browse tags and Recently used", async () => {
    const { result } = renderKeywords()
    await waitFor(() => {
      expect(result.current.availableKeywords).toEqual(["thesis", "reading"])
    })

    act(() => {
      result.current.handleKeywordFilterChange(["thes", "thesis"])
    })

    expect(result.current.keywordTokens).toEqual(["thes", "thesis"])
    expect(result.current.availableKeywords).not.toContain("thes")
    expect(result.current.recentKeywordPickerOptions).toEqual(["thesis"])
  })

  it("still lists a notebook's tags", async () => {
    const { result } = renderKeywords(["chapter-1"])
    await waitFor(() => {
      expect(result.current.availableKeywords).toEqual(["thesis", "reading", "chapter-1"])
    })
  })
})
