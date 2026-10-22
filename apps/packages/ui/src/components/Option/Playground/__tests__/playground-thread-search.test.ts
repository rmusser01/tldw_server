// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest"

import {
  collectThreadSearchMatches,
  getWrappedMatchIndex,
  normalizeThreadSearchQuery,
  revealThreadSearchMark,
  threadSearchHighlightQuery
} from "../playground-thread-search"

describe("playground thread search helpers", () => {
  it("normalizes query by trimming and lowercasing", () => {
    expect(normalizeThreadSearchQuery("  HeLLo  ")).toBe("hello")
  })

  it("collects case-insensitive message matches", () => {
    const matches = collectThreadSearchMatches(
      [
        { message: "Hello world" },
        { message: "second turn" },
        { message: "HELLO again" },
        { message: null }
      ],
      "hello"
    )

    expect(matches).toEqual([0, 2])
  })

  it("returns no matches for blank query", () => {
    expect(collectThreadSearchMatches([{ message: "test" }], "   ")).toEqual([])
  })

  it("wraps next/previous match indexes", () => {
    expect(getWrappedMatchIndex(0, 3, 1)).toBe(1)
    expect(getWrappedMatchIndex(2, 3, 1)).toBe(0)
    expect(getWrappedMatchIndex(0, 3, -1)).toBe(2)
  })
})

describe("thread search phrase and tall-message helpers (CS-11)", () => {
  it("matches a phrase across line breaks and repeated spaces", () => {
    expect(
      collectThreadSearchMatches(
        [{ message: "see Part\n17 now" }, { message: "Part 2 and 17" }],
        "part   17"
      )
    ).toEqual([0])
  })

  it("highlights the typed phrase only while the search bar is open", () => {
    expect(threadSearchHighlightQuery(true, "  Part 17 ")).toBe('"Part 17"')
    expect(threadSearchHighlightQuery(false, "Part 17")).toBe("")
    expect(threadSearchHighlightQuery(true, "   ")).toBe("")
  })

  const buildTranscript = (messageHeight: number) => {
    const container = document.createElement("div")
    Object.defineProperty(container, "clientHeight", { value: 500 })
    const message = document.createElement("div")
    message.dataset.testid = "chat-message"
    message.dataset.index = "3"
    message.getBoundingClientRect = () =>
      ({ height: messageHeight, top: 0, bottom: messageHeight }) as DOMRect
    const mark = document.createElement("mark")
    const scroll = vi.fn()
    mark.scrollIntoView = scroll
    message.append(mark)
    container.append(message)
    return { container, scroll }
  }

  it("scrolls to the first highlight inside a message taller than the transcript", () => {
    const { container, scroll } = buildTranscript(2000)
    expect(revealThreadSearchMark(container, 3, "auto")).toBe(true)
    expect(scroll).toHaveBeenCalledWith({ block: "center", behavior: "auto" })
  })

  it("leaves a message that fits on screen centred as a whole", () => {
    const { container, scroll } = buildTranscript(200)
    expect(revealThreadSearchMark(container, 3, "auto")).toBe(false)
    expect(scroll).not.toHaveBeenCalled()
  })
})
