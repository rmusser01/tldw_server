import React from "react"
import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { UnifiedLoadingState } from "../UnifiedLoadingState"

const { tSpy } = vi.hoisted(() => ({
  tSpy: vi.fn(
    (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  )
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: tSpy
  })
}))

afterEach(() => {
  tSpy.mockClear()
  vi.restoreAllMocks()
})

describe("UnifiedLoadingState", () => {
  it("adapts active sources to the canonical LoadingState primitive", () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined)
    const { container } = render(
      <UnifiedLoadingState
        sources={[
          { key: "local", loading: true, label: "Local data" },
          { key: "server", loading: false, label: "Server sync" },
          { key: "folders", loading: true }
        ]}
        showLabels
      >
        <div>Loaded content</div>
      </UnifiedLoadingState>
    )

    expect(
      container.querySelector('[data-ds-component="LoadingState"]')
    ).toBeInTheDocument()
    expect(screen.getByText("Local data")).toBeInTheDocument()
    expect(screen.queryByText("Server sync")).not.toBeInTheDocument()
    expect(screen.getByText("Loading: folders")).toBeInTheDocument()
    expect(screen.queryByText("Loaded content")).not.toBeInTheDocument()
  })

  it("renders children once all sources finish loading", () => {
    const { container } = render(
      <UnifiedLoadingState
        sources={[
          { key: "local", loading: false, label: "Local data" },
          { key: "server", loading: false, label: "Server sync" }
        ]}
        showLabels
      >
        <div>Loaded content</div>
      </UnifiedLoadingState>
    )

    expect(screen.getByText("Loaded content")).toBeInTheDocument()
    expect(
      container.querySelector('[data-ds-component="LoadingState"]')
    ).not.toBeInTheDocument()
    expect(screen.queryByText("Local data")).not.toBeInTheDocument()
  })

  it("does not translate source labels when labels are hidden", () => {
    render(
      <UnifiedLoadingState
        sources={[{ key: "folders", loading: true }]}
        showLabels={false}
      />
    )

    expect(tSpy).not.toHaveBeenCalled()
    expect(screen.queryByText("Loading: folders")).not.toBeInTheDocument()
  })

  it("keeps the development warning for active sources without labels", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined)

    render(
      <UnifiedLoadingState
        sources={[{ key: "folders", loading: true }]}
        showLabels
      />
    )

    expect(warn).toHaveBeenCalledWith(
      "[UnifiedLoadingState] Missing labels for loading sources:",
      ["folders"]
    )
  })
})
