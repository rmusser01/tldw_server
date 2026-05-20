// @vitest-environment jsdom

import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

const getDesignSystemStateLabelMock = vi.hoisted(() =>
  vi.fn((key: string, fallback: string) => {
    if (key === "ready") return "Registry ready"
    if (key === "blocked") return "Registry blocked"
    return fallback
  })
)

vi.mock("@/design-system/states", () => ({
  getDesignSystemStateLabel: getDesignSystemStateLabelMock
}))

import { AudioReadinessStrip } from "../AudioReadinessStrip"

describe("AudioReadinessStrip", () => {
  it("renders accessible readiness status labels from the state registry", () => {
    render(
      <AudioReadinessStrip
        label="STT readiness"
        items={[
          {
            id: "models",
            label: "STT models",
            state: "ready",
            detail: "2 listed, 1 ready, 1 unknown",
            source: "health"
          },
          {
            id: "blocked",
            label: "Streaming",
            state: "blocked",
            detail: "Missing API credentials",
            source: "provider"
          }
        ]}
      />
    )

    expect(screen.getByRole("status", { name: "STT readiness" })).toHaveTextContent(
      "STT models: Registry ready"
    )
    expect(screen.getByRole("status", { name: "STT readiness" })).toHaveTextContent(
      "Streaming: Registry blocked"
    )
    expect(
      screen.getByLabelText(
        "STT models: Registry ready. 2 listed, 1 ready, 1 unknown Source: model health."
      )
    ).toBeInTheDocument()
    expect(
      screen.getByLabelText(
        "Streaming: Registry blocked. Missing API credentials Source: provider metadata."
      )
    ).toBeInTheDocument()
  })
})
