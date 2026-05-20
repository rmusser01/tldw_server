// @vitest-environment jsdom

import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

vi.mock("@/design-system/states", () => ({
  READY_STATE_LABEL: "Registry ready",
  DEGRADED_STATE_LABEL: "Registry degraded",
  BLOCKED_STATE_LABEL: "Registry blocked"
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
          },
          {
            id: "warning",
            label: "Server catalog",
            state: "warning",
            detail: "Server returned partial metadata",
            source: "response_schema"
          },
          {
            id: "unknown",
            label: "Provider",
            state: "unknown",
            detail: "No provider metadata yet",
            source: "unknown"
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
    expect(screen.getByRole("status", { name: "STT readiness" })).toHaveTextContent(
      "Server catalog: Registry degraded"
    )
    expect(screen.getByRole("status", { name: "STT readiness" })).toHaveTextContent(
      "Provider: Unknown"
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
    expect(
      screen.getByLabelText(
        "Server catalog: Registry degraded. Server returned partial metadata Source: response schema."
      )
    ).toBeInTheDocument()
    expect(
      screen.getByLabelText(
        "Provider: Unknown. No provider metadata yet Source: unknown source."
      )
    ).toBeInTheDocument()
  })
})
