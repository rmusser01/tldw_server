import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  type SourceReviewHandoffPayload,
  buildSourceReviewFlashcardsIntent,
  buildSourceReviewQuizRoute,
  buildSourceReviewRereadContent,
  getSourceReviewItems,
  loadSourceReviewHandoff,
  saveSourceReviewHandoff
} from "../source-review-handoff"

const payload = (): SourceReviewHandoffPayload => ({
  occurrence_id: 31,
  plan_id: 7,
  plan_title: "Cardiac physiology",
  activity_type: "quiz",
  source_bundle: {
    items: [
      {
        source_type: "note",
        source_id: "note-42",
        label: "Cardiac physiology",
        excerpt_text: "Frank-Starling mechanism",
        locator: { section: "Hemodynamics" }
      }
    ]
  }
})

describe("source review handoff helpers", () => {
  beforeEach(() => {
    sessionStorage.clear()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("derives source items and readable snapshot content", () => {
    const handoff = payload()

    expect(getSourceReviewItems(handoff)).toEqual(handoff.source_bundle.items)
    expect(buildSourceReviewRereadContent(handoff)).toContain(
      "Frank-Starling mechanism"
    )
    expect(buildSourceReviewRereadContent(handoff)).toContain("Hemodynamics")
  })

  it("builds bounded flashcards and cloze generation intent without a URL payload", () => {
    const handoff = payload()
    handoff.activity_type = "cloze"
    handoff.source_bundle.items[0].excerpt_text = "x".repeat(30_000)

    const intent = buildSourceReviewFlashcardsIntent(handoff)

    expect(intent.activity_type).toBe("cloze")
    expect(intent.text.length).toBeLessThanOrEqual(20_000)
    expect(intent.source_items).toEqual(handoff.source_bundle.items)
  })

  it("shares the generation text budget across every source", () => {
    const handoff = payload()
    handoff.activity_type = "flashcards"
    handoff.source_bundle.items = [
      {
        source_type: "note",
        source_id: "first",
        label: "First source",
        excerpt_text: "a".repeat(20_000)
      },
      {
        source_type: "media",
        source_id: "second",
        label: "Second source",
        excerpt_text: "b".repeat(20_000)
      }
    ]

    const intent = buildSourceReviewFlashcardsIntent(handoff)

    expect(intent.text.length).toBeLessThanOrEqual(20_000)
    expect(intent.text).toContain("First source")
    expect(intent.text).toContain("Second source")
    expect(intent.text).toContain("a".repeat(100))
    expect(intent.text).toContain("b".repeat(100))
  })

  it("keeps every source represented when labels exceed the text budget", () => {
    const handoff = payload()
    handoff.source_bundle.items = [
      {
        source_type: "note",
        source_id: "first",
        label: `First source ${"x".repeat(25_000)}`
      },
      {
        source_type: "media",
        source_id: "second",
        label: `Second source ${"y".repeat(25_000)}`
      }
    ]

    const intent = buildSourceReviewFlashcardsIntent(handoff)

    expect(intent.text.length).toBeLessThanOrEqual(20_000)
    expect(intent.text).toContain("First source")
    expect(intent.text).toContain("Second source")
  })

  it("stores quiz payload behind a short token and never exposes excerpts in the URL", () => {
    const handoff = payload()

    const route = buildSourceReviewQuizRoute(handoff)
    const token = new URL(route, "https://local.test").searchParams.get(
      "source_review_token"
    )

    expect(route).toMatch(
      /^\/quiz\?tab=generate&source_review=1&source_review_token=/
    )
    expect(route).not.toContain("Frank-Starling")
    expect(token).toBeTruthy()
    expect(loadSourceReviewHandoff(token!)).toEqual(handoff)
    expect(loadSourceReviewHandoff(token!)).toBeNull()
  })

  it("can inspect a handoff without consuming it during render", () => {
    const handoff = payload()
    const route = buildSourceReviewQuizRoute(handoff)
    const token = new URL(route, "https://example.test").searchParams.get(
      "source_review_token"
    )

    expect(loadSourceReviewHandoff(token!, false)).toEqual(handoff)
    expect(loadSourceReviewHandoff(token!)).toEqual(handoff)
    expect(loadSourceReviewHandoff(token!)).toBeNull()
  })

  it("prunes expired handoffs before storing a new payload", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-07-09T12:00:00Z"))
    sessionStorage.setItem(
      "tldw:source-review-handoff:expired",
      JSON.stringify({ expires_at: Date.now() - 1, payload: payload() })
    )
    sessionStorage.setItem("unrelated", "keep")

    expect(saveSourceReviewHandoff(payload())).toBeTruthy()
    expect(
      sessionStorage.getItem("tldw:source-review-handoff:expired")
    ).toBeNull()
    expect(sessionStorage.getItem("unrelated")).toBe("keep")
  })

  it("returns null for missing or expired tokens without throwing", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-07-09T12:00:00Z"))
    const token = saveSourceReviewHandoff(payload())
    vi.advanceTimersByTime(31 * 60 * 1000)

    expect(loadSourceReviewHandoff("missing")).toBeNull()
    expect(loadSourceReviewHandoff(token)).toBeNull()
  })

  it("falls back to the Quiz generate tab when session storage is unavailable", () => {
    const original = window.sessionStorage
    const unavailableStorage: Storage = {
      length: 0,
      clear: vi.fn(),
      getItem: vi.fn(() => null),
      key: vi.fn(() => null),
      removeItem: vi.fn(),
      setItem: vi.fn(() => {
        throw new Error("storage unavailable")
      })
    }
    Object.defineProperty(window, "sessionStorage", {
      configurable: true,
      value: unavailableStorage
    })

    try {
      expect(buildSourceReviewQuizRoute(payload())).toBe(
        "/quiz?tab=generate&source_review=1"
      )
    } finally {
      Object.defineProperty(window, "sessionStorage", {
        configurable: true,
        value: original
      })
    }
  })

  it("returns null when reading and cleaning session storage both fail", () => {
    const original = window.sessionStorage
    const unavailableStorage: Storage = {
      length: 0,
      clear: vi.fn(),
      getItem: vi.fn(() => {
        throw new Error("read unavailable")
      }),
      key: vi.fn(() => null),
      removeItem: vi.fn(() => {
        throw new Error("cleanup unavailable")
      }),
      setItem: vi.fn()
    }
    Object.defineProperty(window, "sessionStorage", {
      configurable: true,
      value: unavailableStorage
    })

    try {
      expect(loadSourceReviewHandoff("unavailable")).toBeNull()
    } finally {
      Object.defineProperty(window, "sessionStorage", {
        configurable: true,
        value: original
      })
    }
  })
})
