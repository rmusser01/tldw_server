// @vitest-environment jsdom

import React from "react"
import { act, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { TFunction } from "i18next"
import RefreshedAtLabel from "../RefreshedAtLabel"

/** Minimal stand-in for the page's `t` — returns the English fallback. */
const t = ((key: string, fallback?: string) => fallback ?? key) as unknown as TFunction

describe("RefreshedAtLabel (admin perf C-S1)", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("test_refreshed_at_label_formats_buckets", () => {
    const now = Date.now()
    vi.setSystemTime(now)

    const { rerender } = render(
      <RefreshedAtLabel at={new Date(now - 5_000)} t={t} />
    )
    expect(screen.getByText(/just now/)).toBeTruthy()

    rerender(<RefreshedAtLabel at={new Date(now - 45_000)} t={t} />)
    expect(screen.getByText(/45s ago/)).toBeTruthy()

    rerender(<RefreshedAtLabel at={new Date(now - 95_000)} t={t} />)
    expect(screen.getByText(/1m ago/)).toBeTruthy()
  })

  it("updates its own text when the 10s tick fires", () => {
    const now = Date.now()
    vi.setSystemTime(now)

    render(<RefreshedAtLabel at={new Date(now)} t={t} />)
    expect(screen.getByText(/just now/)).toBeTruthy()

    act(() => {
      vi.advanceTimersByTime(10_000)
    })
    expect(screen.getByText(/10s ago/)).toBeTruthy()
  })

  it("renders nothing before the first refresh lands", () => {
    vi.setSystemTime(Date.now())

    const { container } = render(<RefreshedAtLabel at={null} t={t} />)
    expect(container.textContent).toBe("")
  })
})
