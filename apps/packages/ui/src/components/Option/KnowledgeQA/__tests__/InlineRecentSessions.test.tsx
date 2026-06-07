import { beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import React from "react"

vi.mock("@/libs/utils", () => ({
  cn: (...args: unknown[]) => args.filter(Boolean).join(" "),
}))

import type { SearchHistoryItem } from "../types"
import { InlineRecentSessions } from "../empty/InlineRecentSessions"

function makeItem(overrides: Partial<SearchHistoryItem> = {}): SearchHistoryItem {
  return {
    id: "item-1",
    query: "test query",
    timestamp: new Date().toISOString(),
    sourcesCount: 3,
    hasAnswer: false,
    ...overrides,
  }
}

describe("InlineRecentSessions", () => {
  beforeEach(() => {
    vi.useRealTimers()
    cleanup()
  })

  it("returns null when items is empty", () => {
    const { container } = render(
      <InlineRecentSessions items={[]} onRestore={vi.fn()} />
    )
    expect(container.innerHTML).toBe("")
  })

  it("renders 'Recent searches' heading when items are provided", () => {
    render(
      <InlineRecentSessions items={[makeItem()]} onRestore={vi.fn()} />
    )
    expect(screen.getByText("Recent searches")).toBeInTheDocument()
  })

  it("renders up to 5 items max when more are provided", () => {
    const items = Array.from({ length: 8 }, (_, i) =>
      makeItem({ id: `item-${i}`, query: `query ${i}` })
    )
    render(
      <InlineRecentSessions items={items} onRestore={vi.fn()} />
    )
    const buttons = screen.getAllByRole("button")
    expect(buttons).toHaveLength(5)
  })

  it("shows query text for each item", () => {
    const items = [
      makeItem({ id: "a", query: "alpha query" }),
      makeItem({ id: "b", query: "beta query" }),
    ]
    render(
      <InlineRecentSessions items={items} onRestore={vi.fn()} />
    )
    expect(screen.getByText("alpha query")).toBeInTheDocument()
    expect(screen.getByText("beta query")).toBeInTheDocument()
  })

  it("shows sourcesCount for each item", () => {
    const items = [
      makeItem({ id: "a", sourcesCount: 7 }),
      makeItem({ id: "b", sourcesCount: 12 }),
    ]
    render(
      <InlineRecentSessions items={items} onRestore={vi.fn()} />
    )
    expect(screen.getByText("7")).toBeInTheDocument()
    expect(screen.getByText("12")).toBeInTheDocument()
  })

  it("shows compact trust label when an item has trust state", () => {
    const item = makeItem({
      hasAnswer: true,
      trustState: "unsynced_local_result",
    })

    render(
      <InlineRecentSessions items={[item]} onRestore={vi.fn()} />
    )

    expect(screen.getByText("Unsynced local result")).toBeInTheDocument()
  })

  it("shows sparkle icon only for items with hasAnswer=true", () => {
    const items = [
      makeItem({ id: "with-answer", query: "answered", hasAnswer: true }),
      makeItem({ id: "no-answer", query: "unanswered", hasAnswer: false }),
    ]
    render(
      <InlineRecentSessions items={items} onRestore={vi.fn()} />
    )
    const buttons = screen.getAllByRole("button")
    // The button for the answered item should contain a Sparkles icon (rendered as an svg)
    const answeredButton = buttons.find((b) =>
      b.textContent?.includes("answered")
    )!
    const unansweredButton = buttons.find((b) =>
      b.textContent?.includes("unanswered")
    )!

    // Sparkles renders as an SVG with the lucide-sparkles class
    // Count SVGs in each button: FileText + Clock3 = 2 without answer, + Sparkles = 3 with answer
    const answeredSvgs = answeredButton.querySelectorAll("svg")
    const unansweredSvgs = unansweredButton.querySelectorAll("svg")
    expect(answeredSvgs.length).toBe(3) // FileText + Sparkles + Clock3
    expect(unansweredSvgs.length).toBe(2) // FileText + Clock3
  })

  it("calls onRestore with the correct item when clicked", async () => {
    const user = userEvent.setup()
    const onRestore = vi.fn()
    const item = makeItem({ id: "click-me", query: "clickable" })
    render(
      <InlineRecentSessions items={[item]} onRestore={onRestore} />
    )
    await user.click(screen.getByRole("button"))
    expect(onRestore).toHaveBeenCalledTimes(1)
    expect(onRestore).toHaveBeenCalledWith(item)
  })

  describe("relative time formatting", () => {
    beforeEach(() => {
      vi.useFakeTimers()
    })

    it("shows 'Just now' for timestamps less than a minute ago", () => {
      vi.setSystemTime(new Date("2026-03-06T12:00:00Z"))
      const item = makeItem({ timestamp: "2026-03-06T12:00:00Z" })
      render(
        <InlineRecentSessions items={[item]} onRestore={vi.fn()} />
      )
      expect(screen.getByText("Just now")).toBeInTheDocument()
    })

    it("shows minutes ago for timestamps under an hour", () => {
      vi.setSystemTime(new Date("2026-03-06T12:15:00Z"))
      const item = makeItem({ timestamp: "2026-03-06T12:00:00Z" })
      render(
        <InlineRecentSessions items={[item]} onRestore={vi.fn()} />
      )
      expect(screen.getByText("15m ago")).toBeInTheDocument()
    })

    it("shows hours ago for timestamps under a day", () => {
      vi.setSystemTime(new Date("2026-03-06T15:00:00Z"))
      const item = makeItem({ timestamp: "2026-03-06T12:00:00Z" })
      render(
        <InlineRecentSessions items={[item]} onRestore={vi.fn()} />
      )
      expect(screen.getByText("3h ago")).toBeInTheDocument()
    })

    it("shows 'Yesterday' for timestamps 1 day ago", () => {
      vi.setSystemTime(new Date("2026-03-07T12:00:00Z"))
      const item = makeItem({ timestamp: "2026-03-06T12:00:00Z" })
      render(
        <InlineRecentSessions items={[item]} onRestore={vi.fn()} />
      )
      expect(screen.getByText("Yesterday")).toBeInTheDocument()
    })

    it("shows days ago for timestamps under a week", () => {
      vi.setSystemTime(new Date("2026-03-10T12:00:00Z"))
      const item = makeItem({ timestamp: "2026-03-06T12:00:00Z" })
      render(
        <InlineRecentSessions items={[item]} onRestore={vi.fn()} />
      )
      expect(screen.getByText("4d ago")).toBeInTheDocument()
    })

    it("shows formatted date for timestamps a week or more ago", () => {
      vi.setSystemTime(new Date("2026-03-20T12:00:00Z"))
      const item = makeItem({ timestamp: "2026-03-06T12:00:00Z" })
      render(
        <InlineRecentSessions items={[item]} onRestore={vi.fn()} />
      )
      // toLocaleDateString with { month: "short", day: "numeric" } produces e.g. "Mar 6"
      expect(screen.getByText(/Mar\s*6/)).toBeInTheDocument()
    })
  })
})
