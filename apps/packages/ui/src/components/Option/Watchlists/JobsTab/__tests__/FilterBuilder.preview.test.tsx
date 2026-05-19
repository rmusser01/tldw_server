// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { FilterBuilder } from "../FilterBuilder"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string },
      maybeOptions?: Record<string, unknown>
    ) => {
      if (typeof fallbackOrOptions === "string") {
        if (!maybeOptions) return fallbackOrOptions
        return fallbackOrOptions.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
          const value = maybeOptions[token]
          return value == null ? "" : String(value)
        })
      }
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        const maybeDefault = fallbackOrOptions.defaultValue
        if (typeof maybeDefault === "string") return maybeDefault
      }
      return key
    }
  })
}))

describe("FilterBuilder preview panel", () => {
  beforeEach(() => {
    if (!window.matchMedia) {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }
  })

  it("renders loading and unavailable fallback states", () => {
    const { rerender } = render(
      <FilterBuilder
        value={[]}
        onChange={vi.fn()}
        preview={{ loading: true }}
      />
    )

    expect(screen.getByTestId("filter-preview-panel")).toBeInTheDocument()
    expect(screen.getByText("Loading sample candidates...")).toBeInTheDocument()

    rerender(
      <FilterBuilder
        value={[]}
        onChange={vi.fn()}
        preview={{ unavailableReason: "Save this monitor first." }}
      />
    )

    expect(screen.getByText("Save this monitor first.")).toBeInTheDocument()
  })

  it("renders empty preview outcome state when no sample candidates are available", () => {
    render(
      <FilterBuilder
        value={[]}
        onChange={vi.fn()}
        preview={{
          outcome: {
            items: [],
            total: 0,
            ingestable: 0,
            filtered: 0
          }
        }}
      />
    )

    expect(screen.getByText("Found 0 articles (0 skipped) out of 0 sample items.")).toBeInTheDocument()
    expect(screen.getByText("No sample candidates available.")).toBeInTheDocument()
  })

  it("shows per-rule summaries and regex validation hints", () => {
    render(
      <FilterBuilder
        value={[
          {
            type: "regex",
            action: "exclude",
            value: { pattern: "(", field: "title", flags: "i" },
            is_active: true
          }
        ]}
        onChange={vi.fn()}
      />
    )

    expect(screen.getByTestId("filter-rule-summary-0")).toHaveTextContent("exclude")
    expect(screen.getByTestId("filter-regex-error-0")).toHaveTextContent("Regex pattern is invalid")
    expect(screen.getByText("Examples: sponsored|promo, (?i)breaking, \\bAI\\b")).toBeInTheDocument()
  })

  it("exposes preview impact action when preview data is available", () => {
    render(
      <FilterBuilder
        value={[]}
        onChange={vi.fn()}
        preview={{
          outcome: {
            items: [],
            total: 0,
            ingestable: 0,
            filtered: 0
          }
        }}
      />
    )

    const previewButton = screen.getByTestId("filter-preview-impact-button")
    expect(previewButton).toBeInTheDocument()
    fireEvent.click(previewButton)
    expect(screen.getByTestId("filter-preview-panel")).toBeInTheDocument()
  })
})
